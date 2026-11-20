"""Settings read from the environment (`.env`)."""

import os
import warnings

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "openai:gpt-5.4-mini")

# langchain-openai calls model_dump() on streamed chunks of a structured response, whose `parsed`
# field the openai SDK types as None: pydantic warns although the output is correct. Library noise.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
