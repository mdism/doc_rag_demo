import os
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from tools.retriever_tool import DocumentRetrieverTool

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
if not OLLAMA_BASE_URL.startswith(("http://", "https://")):
    OLLAMA_BASE_URL = "http://" + OLLAMA_BASE_URL

DEFAULT_AGENT_LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "ollama/llama3.1:8b")


def build_llm(model_name: str) -> LLM:
    """
    Builds an LLM object for the given model name. Called fresh per crew
    instance so the model can be chosen per-request (e.g. from OpenWebUI's
    model picker) rather than being fixed once at import time.
    """
    if "gemma3" in model_name:
        import litellm
        litellm.register_model({model_name: {"supports_function_calling": False}})

    return LLM(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        extra_headers={"ngrok-skip-browser-warning": "true"},
        timeout=300,
    )

@CrewBase
class DocRagDemoCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, llm_model: str = None):
        self.llm = build_llm(llm_model or DEFAULT_AGENT_LLM_MODEL)

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            llm=self.llm,
            tools=[DocumentRetrieverTool()],
            max_iter=5,
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["answer_synthesizer"],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
        )

    @task
    def search_the_knowledge_base_task(self) -> Task:
        return Task(config=self.tasks_config["search_the_knowledge_base_task"])

    @task
    def using_the_retrieved_passages_task(self) -> Task:
        return Task(
            config=self.tasks_config["using_the_retrieved_passages_task"],
            context=[self.search_the_knowledge_base_task()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )   