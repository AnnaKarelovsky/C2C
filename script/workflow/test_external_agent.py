"""Test ExternalToolAgent with search tool."""

from dotenv import find_dotenv, load_dotenv
from camel.toolkits import FunctionTool, SearchToolkit
from rosetta.workflow.retriever import search_engine
from rosetta.workflow.camel_utils import ExternalToolAgent, create_model
from rosetta.workflow.display import ConvLogger

load_dotenv(find_dotenv())

# Configuration
model = create_model(
    "fireworks",
    model_type="accounts/fireworks/models/gpt-oss-120b",
    temperature=0.0,
    max_tokens=4096
)

tools = [FunctionTool(search_engine)]
question = "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"

if __name__ == "__main__":
    # Create logger for live message display (transient=True clears when done)
    logger = ConvLogger(max_messages=4, max_content_len=200, transient=True)

    agent = ExternalToolAgent(
        system_message="You are a helpful research assistant. Use the search and get_document tools to find information.",
        model=model,
        tools=tools,
        reserved_tokens=2048,
        token_limit=128000,  # Context window size (not output tokens)
        logger=logger,
    )

    print("Question:", question)
    print("=" * 50)

    result = agent.step(question, max_iterations=50)

    print("\n" + "=" * 50)
    print("Final Answer:")
    print(result.content)
    print("\n" + "=" * 50)
    print(f"Tool calls: {result.num_tool_calls}")
    print(f"Tools used: {result.tools_used}")
    print(f"Terminated early: {result.terminated_early}")
    if result.termination_reason:
        print(f"Termination reason: {result.termination_reason}")
