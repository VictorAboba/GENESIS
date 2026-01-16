from agent import AutonomousAgent
from dotenv import load_dotenv
import os
import shutil

from openai import OpenAI


def prepare_env():
    with open("/app/src/tool_registry.json", "w") as file:
        file.write("{}")
    with open("/app/src/notes.json", "w") as file:
        file.write("{}")

    tools_dir = "/app/src/created_tools"

    if os.path.exists(tools_dir):
        for item in os.listdir(tools_dir):
            if item == "__init__.py":
                continue

            item_path = os.path.join(tools_dir, item)

            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")


if __name__ == "__main__":
    load_dotenv()
    prepare_env()
    API_KEY = os.environ["API_KEY"]
    BASE_URL = os.environ["BASE_URL"]
    MODEL = os.environ["MODEL"]
    HF_MODEL = os.environ["HF_MODEL"]
    CONTEXT_WINDOW = int(os.environ["CONTEXT_WINDOW"])

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    agent = AutonomousAgent(
        hf_model=HF_MODEL,
        model=MODEL,
        max_context_tokens=CONTEXT_WINDOW,
        client=client,
    )
    agent.live()
