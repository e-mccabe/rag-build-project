"""Splitting corupus documents into small sections with sufficient content and semantic meaning"""

from dataclasses import dataclass
from rag_build.loading import Document

import re

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
            # Select text after the has signs
            heading_text = heading_match.group(2).strip()
            # If a heading is hit then take all the take in compiled text as the previous section
            text = "\n".join(compiled_text).strip()
            
            if text:
                sections.append((list(heading_stack),text))
                # Clear all the text so it is clean for the next section
                compiled_text.clear()

            # Calculate the heading level based on the number of # Characters
            heading_level = len(heading_match.group(1))
            # Delete any remaining n level headings or deeper headings from the previous section
            del heading_stack[heading_level - 1:]
            # The vacated slot at index n-1 now holds this heading, ancestors above it stay in scope
            heading_stack.append(heading_text)
            
        else:
            compiled_text.append(line)

    # Compile anything remaining in compiled text to a full piece of text
    text = "\n".join(compiled_text).strip()
    if text or heading_stack:
        sections.append((heading_stack,text))

    return sections
            

def chunking_document(document:Document) -> list[Chunk]:
    """Building and indexing the chunk dataclass including the content, metadata and breadcrumb to the section"""
    chunks: list[Chunk] = []
    document_chunk_index: int  = 0 

    for headings, section in _split_string_by_headers(document.content):

        chunks.append(
            Chunk(
                text = f"\n{section}\n",
                metadata={
                    "source":document.path,
                    "file":document.name,
                    "headings":headings,
                    "index":document_chunk_index,
                    **document.metadata
                }
                )
                    )
        document_chunk_index += 1
    
    return chunks

def chunk_all_documents(documents:list[Document]) -> list[Chunk]:
    """Full chunking of the whole corpus of .md documents"""
    chunked_documents: list[Chunk] = []

    for document in documents:
        chunked_documents.extend(chunking_document(document))
    
    return chunked_documents








