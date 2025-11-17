import asyncio
import uuid

from google.adk import Runner
from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tools_generator.generator.agent_generator import AgentGenerator
from google.adk.tools.agent_tool import AgentTool

from utils import load_from_file


state_data = {
    "org_name": "",
    "base_url": "",
    "swagger_json": "",
}

def set_org_name(org_name: str) -> None:
    """
    Set the organization name in the state data.

    Args:
        org_name (str): The name of the organization to set.

    Returns:
        None
    """
    state_data["org_name"] = org_name
    print(f"Organization name set to: {org_name}")


def set_base_url(base_url: str) -> None:
    """
    Set the base URL of the API in the state data.

    Args:
        base_url (str): The base URL to set.

    Returns:
        None
    """
    state_data["base_url"] = base_url
    print(f"Base URL set to: {base_url}")


def set_swagger_json(swagger_json: str) -> None:
    """
    Set the Swagger JSON specification in the state data.

    Args:
        swagger_json (str): The Swagger JSON specification as a string.

    Returns:
        None
    """
    state_data["swagger_json"] = swagger_json
    print("Swagger JSON specification set.")

def validate_state() -> bool:
    """
    Validate whether the required state data has been properly set.
    Validation checks that 'org_name', 'base_url', and 'swagger_json' are non-empty strings.

    Returns:
        bool: True if all required fields are non-empty strings, False otherwise.
    """
    is_valid = all(
        isinstance(state_data.get(key), str) and state_data[key].strip() != ""
        for key in ["org_name", "base_url", "swagger_json"]
    )

    if is_valid:
        agent = create_agent()
        res = tool_generation_agent
        # agent.write_to_tool(res)

    return is_valid



def create_agent():
    """
    This function initializes an AgentGenerator object with the organization name
    stored in the global state_data dictionary under the key 'org_name'.

    Returns:
        AgentGenerator: An instance of AgentGenerator initialized with the organization name.
    """
    ag  = AgentGenerator(state_data["org_name"])
    state_data["agent"] = ag

    return ag


def write_code_to_tool(code: str) -> None:
    """
    Writes (appends) the provided Python code string to the dynamically generated
    API tool file of the current agent.

    This function is intended to be used by the tool-generation pipeline after
    the Swagger/OpenAPI specification has been fully processed and validated.
    It delegates the actual file write operation to the active agent instance
    stored in the shared state.

    Args:
        code (str):
            The complete or incremental Python code (usually a class definition,
            function, or helper) that should be appended to the agent's tool file.
            The string must be valid Python syntax and properly indented.

    Returns:
        None

    Raises:
        KeyError: If ``state_data["agent"]`` is not set (i.e., no agent has been
                  created or loaded yet).
        AttributeError: If the agent object stored in state does not implement
                        the ``write_to_tool`` method.
        OSError: Propagated from the underlying file I/O operation performed by
                 ``agent.write_to_tool``.

    Example:
        # Appends the class definition to the agent's tool Python file.
    """
    state_data["agent"].write_to_tool(code)


tool_generation_agent = Agent(
    model='gemini-2.5-flash',
    name='tool_generation_agent',
    description='An agent that generates Python tools on the fly for REST API calls from Swagger JSON.',
    instruction=(
        "You are a developer assistant whose main job is to generate dynamic Python tool functions for REST API endpoints, "
        "using Swagger (OpenAPI) JSON as input. state['swagger_json']"
        "Each tool should contain all needed parameters and logic to make HTTP requests as defined by the Swagger spec, "
        "including handling authentication, query/body parameters, and error handling. "
        "When you receive a Swagger JSON, parse it to create one or more Python functions under a 'tools.py' file, "
        "making sure each function matches its corresponding API operation."
        "and host with base url will be given to you state['base_url']"
        "once all coding is made add those in an list with variable name 'tool_list' eg) tool_list = [example_tool]"
        "Do not answer general knowledge questions. Always output code and descriptions focused on creating these tools."
        "return only pure python code."
    )
)

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='Strict orchestrator for dynamic REST API tool generation from Swagger/OpenAPI specs.',
    instruction=(
        "You are a STRICT orchestrator agent. "
        "You MUST NEVER answer questions, write code, or interact with the user directly except for short coordination messages. "
        "All actions are performed exclusively via tool calls.\n\n"

        "Your only responsibilities:\n"
        "1. Ensure all required data is collected via your tools (never ask the user yourself).\n"
        "2. Validate the collected data.\n"
        "3. Trigger the actual tool-generation agent when validation passes.\n\n"

        "Required data (must be set via tool calls only):\n"
        "• Organization / API name → set_org_name\n"
        "• Base URL / host         → set_base_url\n"
        "• Full Swagger/OpenAPI JSON → set_swagger_json\n"
        "• Authentication details (if any) – handled inside the tools or sub-assistant flow\n\n"

        "Exact workflow you MUST follow:\n"
        "1. As soon as the user expresses intent to generate API tools, "
        "   immediately delegate to the sub-assistant (or directly use the setter tools) "
        "   to collect any missing information. Respond only with a short message like "
        "   'Collecting required API information…' if needed.\n\n"
        "2. Once you confirm (via tool results or sub-assistant completion) that all data is present, "
        "   call the tool `validate_state`.\n\n"
        "3. If `validate_state` returns {'valid': True}, "
        "   your VERY NEXT action MUST be a tool call to `create_agent` "
        "   (this spawns the real tool_generation_agent that will handle code writing).\n\n"
        "4. If `validate_state` returns {'valid': False} or lists missing/invalid items, "
        "   delegate back to the sub-assistant (or re-trigger the relevant setter tools) "
        "   to fix the issues. Do not proceed further until validation passes.\n\n"

        "Critical Rules:\n"
        "- NEVER call `write_code_to_tool` yourself — only the tool_generation_agent does that.\n"
        "- NEVER output Python code or Swagger parsing logic.\n"
        "- NEVER store or assume state — rely only on the shared state updated by the setter tools.\n"
        "- After successful validation, you have exactly ONE job: call `create_agent`. "
        "   Do not add extra text or delays.\n"
        "- Keep any human-facing text extremely short and only for coordination "
        "   (e.g., 'Validating collected data…', 'Generating API tools now…').\n\n"

        "You are the gatekeeper — collect → validate → trigger. Nothing else."
    ),
    tools=[
        set_org_name,
        set_base_url,
        set_swagger_json,
        validate_state,
    ]
)

runner = InMemoryRunner(
    agent=tool_generation_agent,
    app_name="simple_app",
)


async def invoke_agent():
    # 1. Set up session and runner (ADK handles multi-step communication and memory)
    session_service = InMemorySessionService()
    runner = Runner(session_service=session_service)

    # 2. Create a session for the user
    user_id = "user_123"
    session = await runner.async_create_session(user_id=user_id, agent_id=root_agent.name)

    # async_stream_query sends the message and streams the response events
    async for event in runner.async_stream_query(
            user_id=user_id,
            session_id=session.id,
            message="Hi"
    ):
        if event.text:
            print(event.text, end="")
    print()  # for a final newline


if __name__ == "__main__":
    # The ADK is built on asyncio
    asyncio.run(invoke_agent())