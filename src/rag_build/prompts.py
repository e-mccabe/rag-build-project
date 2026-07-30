"""Load prompt templates from the prompts/ directory at startup"""
from dataclasses import dataclass

from rag_build.config import PATHS


def _load_prompts(name: str) -> str:
    """Read a prompt, fail with message if it is missing"""
    path = PATHS.prompts / f'{name}.md'
    if not path.exists():
        raise FileNotFoundError(f'Prompt file {name}.md not found in {path}')
    return path.read_text(encoding='utf-8')


# Loading prompts
SYSTEM_PROMPT = _load_prompts('system')
RERANK_PROMPT = _load_prompts('rerank')
SINGLE_HOP_PROMPT = _load_prompts('single_hop_eval')
MULTI_HOP_PROMPT = _load_prompts('multi_hop_eval')

@dataclass(frozen=True)
class Prompts:
    """Prompt identifiers by task"""
    system:     str = SYSTEM_PROMPT
    rerank:     str = RERANK_PROMPT
    single_hop: str = SINGLE_HOP_PROMPT
    multi_hop:  str = MULTI_HOP_PROMPT

PROMPTS = Prompts()