from agent import AutonomousAgent
from dotenv import load_dotenv
import os

from openai import OpenAI

if __name__ == "__main__":
    load_dotenv()
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
