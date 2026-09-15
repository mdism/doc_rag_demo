from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from tools.retriever_tool import DocumentRetrieverTool

# llama3.1:8b has genuine native tool-calling support in Ollama, unlike
# gemma3 (which has none, on any tag). No forced litellm override needed
# here -- native tool calling works correctly for this model out of the box.
# _llm = "ollama/gemma3:latest"
_llm = LLM(
    model="ollama/llama3.1:8b",
    extra_headers={"ngrok-skip-browser-warning": "true"},
    timeout=300,
)
 


@CrewBase
class DocRagDemoCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            llm=_llm,
            tools=[DocumentRetrieverTool()],
            max_iter=5,
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["answer_synthesizer"],
            llm=_llm,
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