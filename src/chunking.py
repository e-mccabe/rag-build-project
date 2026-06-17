"""Split single document into sections with sufficient content and semantic meaning"""

from dataclasses import dataclass
from src.loading import Document

import re

# Defining the constant parameters for chunking
MAX_CHARACTERS = 1500
MIN_CHARACTERS = 100

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")

# Define data structure for each chunk
@dataclass
class Chunk:
    text: str
    metadata : dict


def _split_string_by_headers(content:str) -> list[tuple[list[str],str]]:

    sections: list[tuple[list[str],str]] = []
    heading_stack: list[str] = []
    compiled_text: list[str] = []

    for line in content.splitlines():

        heading_match = HEADING_RE.match(line)
        if heading_match:
            heading_text = heading_match.group(2).strip()
            text = "\n".join(compiled_text).strip()
            
            if text:
                sections.append((list(heading_stack),text))
                compiled_text.clear()
            
            heading_level = len(heading_match.group(1))
            del heading_stack[heading_level - 1:]
            heading_stack.append(heading_text)
            

        else:
            compiled_text.append(line)
        
    text = "\n".join(compiled_text).strip()
    if text or heading_stack:
        sections.append((heading_stack,text))

    return sections
            

def chunking_document(document:Document) -> list[Chunk]:

    chunks: list[Chunk] = []
    index: int  = 0

    for headings, section in _split_string_by_headers(document.content):

        breadcrumbs = ' > '.join([document.name,*headings])
        chunks.append(
            Chunk(
                text = f"# {breadcrumbs}\n\n{section}\n",
                metadata={
                    "source":document.path,
                    "file":document.name,
                    "headings":headings,
                    "index":index
                }
                )
                    )
        index += 1
    
    return chunks

def chunk_all_documents(documents:list[Document]) -> list[Chunk]:

    chunked_documents: list[Chunk] = []

    for document in documents:
        chunked_documents.extend(chunking_document(document))
    
    return chunked_documents










