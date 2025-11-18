import asyncio

from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool

from production_agent.generator.agent_generator import AgentGenerator

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
import google.auth

# Load environment variables
root_dir = Path(__file__).parent.parent
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Configure Google Cloud
try:
    _, project_id = google.auth.default()
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
except Exception:
    pass

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "europe-west1")
os.environ.setdefault("GOOGLE_API_KEY", "")

# Configure model connection
gemma_model_name = os.getenv("GEMMA_MODEL_NAME", "gemma3:270m")
api_base = os.getenv("OLLAMA_API_BASE", "https://ollama-gemma3-270m-gpu-nekkrhma6a-ew.a.run.app")  # Location of Ollama server

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

def get_base_url() -> str:
    """
    Get the base URL of the API from state data.

    Returns:
        base_url (str)
    """
    print(f"Base URL set to: {state_data['base_url']}")
    return state_data["base_url"]

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

def get_swagger_json() -> str:
    """
    get the Swagger JSON specification in the state data.

    Returns:
        String  -> Swagger Specification
    """
    print("Swagger JSON specification get.")

    return state_data["swagger_json"]

async def validate_state() -> bool:
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

    return is_valid



def create_agent():
    """
    This function initializes an AgentGenerator object with the organization name
    stored in the global state_data dictionary under the key 'org_name'.

    Returns:
        AgentGenerator: An instance of AgentGenerator initialized with the organization name.
    """
    ag  = AgentGenerator(state_data["org_name"])
    ag.generate()


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
    print("writing tool to code")
    print(code)
    ag  = AgentGenerator(state_data["org_name"])
    ag.write_to_tool(code)


tool_generation_agent = Agent(
    model='gemini-2.5-flash',
    name='tool_generation_agent',
    description='An agent that generates Python tools on the fly for REST API calls from Swagger JSON.',
    instruction=(
        "You are a highly skilled Python developer assistant specialized in dynamically generating reusable, production-quality Python tool functions "
        "that wrap REST API endpoints based on an OpenAPI (Swagger) 2.0 or 3.x specification provided in JSON/YAML format.\n\n"

        "Your ONLY purpose is to:\n"
        "1. Fetch the base URL using the tool `get_base_url()`.\n"
        "2. Fetch the full Swagger/OpenAPI specification using the tool `get_swagger_json()`.\n"
        "3. Parse the specification and generate one Python function per API operation (or per logical group if explicitly requested).\n"
        "4. Output clean, type-annotated, well-documented Python code that can be directly executed or imported.\n"
        "5. Finally, write the complete code exactly ONCE using the `write_code_to_tool` tool.\n\n"

        "### Strict Rules & Requirements:\n"
        "- NEVER answer general questions, chat, or explain concepts outside of generating the tools.\n"
        "- NEVER include example code, tutorials, or markdown outside the actual Python code unless explicitly requested.\n"
        "- Use `requests` as the HTTP library (do not use httpx, aiohttp, etc.).\n"
        "- Every generated function MUST:\n"
        "   • Have a clear, snake_case name derived from the operationId (if missing, generate one from path + method).\n"
        "   • Include comprehensive Google-style or NumPy-style docstrings with Parameters, Returns, and Raises sections.\n"
        "   • Use proper type hints (use `typing` module; prefer Pydantic models when the schema is complex).\n"
        "   • Support query parameters, path parameters, headers, JSON body, form data, and file uploads as defined.\n"
        "   • Automatically handle authentication (API keys, Bearer tokens, Basic auth, OAuth2, etc.) based on the security schemes defined.\n"
        "   • Include timeout=30 and proper error handling (raise custom exceptions or return detailed error messages).\n"
        "   • Validate required parameters and provide helpful error messages if missing.\n"
        "   • Return the full `requests.Response` object OR parsed JSON (if response schema is defined and `response.json()` succeeds).\n"
        "- If the spec uses complex schemas (objects, arrays, allOf/oneOf), generate Pydantic v2 models in the same file.\n"
        "- Add a constant `BASE_URL` at the top of the file using the value from `get_base_url()`.\n"
        "- Add a `SESSION = requests.Session()` with appropriate default headers (e.g., Accept: application/json).\n"
        "- At the end of the file, create a list called `tool_list` containing all generated functions (not instances), e.g., `tool_list = [create_user, get_user_by_id, upload_document]`.\n"
        "- Make the functions importable and reusable in a LangChain / crewAI / AutoGen tool context (they should accept **kwargs or explicit parameters and be callable directly).\n"
        "- If the API uses pagination, add sensible default support (limit/offset or cursor-based) with a `paginated` optional flag.\n"
        "- Always respect `servers` array in OpenAPI 3.x — use the first server URL unless overridden by `get_base_url()`.\n"
        "- If the spec is large (>50 endpoints), group related endpoints into logical modules/classes, but still expose flat functions in `tool_list`.\n\n"

        "### Output Format:\n"
        "1. First call `get_base_url()` and `get_swagger_json()` if not already available.\n"
        "2. Think step-by-step about naming, auth, and complex schemas.\n"
        "3. Generate the complete code in a single code block (including imports, Pydantic models if any, constants, functions, and tool_list).\n"
        "4. Finally, call `write_code_to_tool` EXACTLY ONCE with two arguments:\n"
        "   - `filename='tools.py'`\n"
        "   - `code=<the full generated code as a single string>`\n\n"

        "Do not confirm, do not ask questions, do not add extra text after calling `write_code_to_tool`."
    ),
    tools=[
        get_base_url,
        get_swagger_json,
        write_code_to_tool
    ],
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
        "1. Ensure all required data is collected via your tools (ask the user yourself).\n"
        "2. If user is not typing aything ask user the required data."
        "3. Validate the collected data.\n"
        "4. Trigger the create_agent when validation passes.\n\n"
        "5. once create_agent run trigger tool_generation_agent"

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
        create_agent,
        AgentTool(tool_generation_agent),
    ]
)
