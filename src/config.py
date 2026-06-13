"""Configuration of project global variables and required key"""
import os
from dotenv import load_dotenv


load_dotenv() # read .env file into the environment

def _require(name:str) -> str:
    """Runs check if environment variable is available. Fails with clear message if not available"""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. {name} needs to be added to the .env file"
        )
    
    return value


OPENAI_API_KEY = _require("OPENAI_API_KEY")

# Project Model Choice 
EMBEDDING_MODEL = 'text-embedding-3-small'

RESPONSE_MODEL = 'gpt-5.4-nano-2026-03-17'