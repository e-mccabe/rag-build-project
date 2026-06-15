"""Loading in the ML Markdown notes"""
from pathlib import Path
from dataclasses import dataclass
import frontmatter

# Defining the structure for each document
@dataclass
class Document:
    path: str
    name: str
    content: str
    metadata: dict


# Build a function that loads in all the .md files
def load_vault(vault_dir:str | Path) -> list[dict]:

    vault = Path(vault_dir)

    documents = []
    for md_path in sorted(vault.rglob('*.md')):

        # Using pathlib to open .md files and extract the relevant YAML frontmatter
        with md_path.open(encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        content = post.content
        metadata = post.metadata
        path = str(md_path.relative_to(vault)).replace("\\",'/')
        name = str(metadata.get('title') or md_path.stem)

        documents.append(
            Document(path = path,
                     name = name,
                     content=content,
                     metadata=metadata)
        )

    return documents


        



