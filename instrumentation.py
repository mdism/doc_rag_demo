import os
from phoenix.otel import register


def setup_tracing(inputs=None):
    """
    Registers this app with the local Phoenix instance and auto-instruments
    any installed OpenInference packages (crewai, litellm) it finds.
    Called once, automatically, right before the crew starts running --
    wired via crew.jsonc's before_kickoff_callbacks.

    CrewAI passes the crew's `inputs` dict into before_kickoff_callbacks
    (so callbacks can inspect/modify inputs); we don't need it here, but
    must accept it and return it unchanged.
    """
    endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
    register(
        project_name="doc-rag-demo",
        endpoint=endpoint,
        auto_instrument=True,
        batch=False,
    )
    print(f"Phoenix tracing enabled -> {endpoint}")
    return inputs


