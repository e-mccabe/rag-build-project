"""Project Configuration: paths, model choices, required secrets"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv() # read .env file into the environment

### ======================================================
###                Helper Functions
### ======================================================
def _require(name:str) -> str:
    """Return a required enivornment variable, or fail with a clear message"""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. {name} needs to be added to the .env file"
        )    
    return value

def _find_root(marker: str = 'pyproject.toml') -> Path:
    """Identify the project root from whatever directory the code is run from"""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent/marker).exists():
            return parent
    raise FileNotFoundError(f'No file: {marker} found above {__file__}')

### ======================================================
###                  Filesystem Paths
### ======================================================
PROJECT_ROOT = _find_root()

@dataclass(frozen=True)
class Paths:
    """Filesystem locations, anchored to project root"""
    corpus  : Path = PROJECT_ROOT / 'data'
    prompts : Path = PROJECT_ROOT / 'prompts'
    eval_dir: Path = PROJECT_ROOT / 'eval_dir'

PATHS = Paths()

### ======================================================
###                  Model Choices
### ======================================================

@dataclass(frozen=True)
class Models:
    """Model identifiers grouped by pipeline role"""
    embedding:  str = 'text-embedding-3-small'
    response:   str = 'gpt-4o-mini'
    reranking:  str = 'gpt-4o'

MODELS = Models()

### ======================================================
###                    API Keys
### ======================================================

@dataclass(frozen=True)
class Keys:
    openai: str = _require("OPENAI_API_KEY")


KEYS = Keys()
