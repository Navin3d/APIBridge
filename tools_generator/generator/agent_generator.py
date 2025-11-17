import os
import subprocess

class AgentGenerator:
    def __init__(self, project: str, path: str = "/Users/navin3d/Programming/Python/Machine-Learning/langchain/Generated_Agents"):
        self.project = project
        self.path = os.path.abspath(path)
        self.project_dir = os.path.join(self.path, self.project)
        self.venv_dir = os.path.join(self.project_dir, "venv")

        self.create_venv()
        self.install_dependencies()
        self.create_agent()
        self.create_tool_file()
        self.map_tool_with_agent()

    def run_subprocess(self, cmd, cwd=None):
        print(f"Running: {cmd}")
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd or self.project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        # Stream output live
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output, end='')  # Already includes newline
        rc = process.poll()
        if rc != 0:
            raise RuntimeError(f"Command failed with return code {rc}: {cmd}")

    def create_venv(self):
        if not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir)
        # Create a Python virtual environment
        print("Creating virtual environment...")
        self.run_subprocess(f"python3 -m venv venv", cwd=self.project_dir)

    def install_dependencies(self):
        # Activate venv and install ADK and other required packages
        print("Installing dependencies...")
        pip_path = os.path.join(self.venv_dir, "bin", "pip")
        self.run_subprocess(f"{pip_path} install google-adk requests")

    def create_agent(self):
        # Run ADK CLI to create the new agent project
        print("Creating ADK agent project...")
        adk_path = os.path.join(self.venv_dir, "bin", "adk")
        command = f"{adk_path} create {self.project} --model gemini-2.5-flash --project resounding-age-478413-s0 --region us-central1"
        self.run_subprocess(command, cwd=self.project_dir)

    def create_tool_file(self):
        tools_path = os.path.join(self.project_dir, self.project, "tools.py")
        os.makedirs(os.path.dirname(tools_path), exist_ok=True)
        tool_code = '''# Agent Generated code
import requests

def example_tool(param1: str) -> dict:
    """
    Example generated tool. Replace logic as needed.
    """
    # Example: This could be a Swagger-based call.
    return {"status": "success", "param1": param1}

tool_list = [example_tool]
        '''
        with open(tools_path, "w") as f:
            f.write(tool_code)
        print(f"Created tool file at {tools_path}")


    def write_to_tool(self, tool_code):
            tools_path = os.path.join(self.project_dir, self.project, "tools.py")
            os.makedirs(os.path.dirname(tools_path), exist_ok=True)
            with open(tools_path, "w") as f:
                f.write(tool_code)
            print(f"Created tool file at {tools_path}")


    def map_tool_with_agent(self):
        agent_path = os.path.join(self.project_dir, self.project, "agent.py")
        # You might want to load and edit this file more robustly, this is a simple example.
        agent_code = '''# Agent Generated code
from google.adk.agents.llm_agent import Agent
from .tools import tool_list

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    tools=tool_list
)
        '''.format(self.project)
        with open(agent_path, "w") as f:
            f.write(agent_code)
        print(f"Mapped tools in agent definition at {agent_path}")


if __name__ == "__main__":
    ag = AgentGenerator("my_example_agent")
