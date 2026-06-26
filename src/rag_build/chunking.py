"""Splitting corupus documents into small sections with sufficient content and semantic meaning"""

from dataclasses import dataclass
from rag_build.loading import Document

import re

# Defining the constant parameters for chunking
MAX_CHARACTERS = 1500 # This currently is used, large paragraph logic to be implemented
MIN_CHARACTERS = 100

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")

# Define data structure for each chunk
@dataclass
class Chunk:
    text: str
    metadata : dict


def _split_string_by_headers(content:str) -> list[tuple[list[str],str]]:
    """Using the # characters in .md headings to split the document into distinct sections"""

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
    """Building and indexing the chunk dataclass including the content, metadata and breadcrumb to the section"""
    chunks: list[Chunk] = []
    index: int  = 0  # Chunk index within a document


    for headings, section in _split_string_by_headers(document.content):

        breadcrumbs = ' > '.join([document.name,*headings])
        chunks.append(
            Chunk(
                text = f"# {breadcrumbs}\n\n{section}\n",
                metadata={
                    "source":document.path,
                    "file":document.name,
                    "headings":headings,
                    "index":index,
                    **document.metadata
                }
                )
                    )
        index += 1
    
    return chunks

def chunk_all_documents(documents:list[Document]) -> list[Chunk]:
    """Full chunking of the whole corpus of .md documents"""
    chunked_documents: list[Chunk] = []

    for document in documents:
        chunked_documents.extend(chunking_document(document))
    
    return chunked_documents










