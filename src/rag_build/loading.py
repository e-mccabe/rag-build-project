"""Loading in the Corpus (ML Markdown notes)"""
# =============================================================================
# MARKER'S NOTE - loading.py
# The cleanest module in the project: one job, done directly, no cleverness.
# Marks lost only on defensive handling (a single malformed file kills the whole
# run) and on the fact that nothing here records *when* a file was last changed,
# which is what forces the expensive re-index downstream. Good foundation.
# =============================================================================
from pathlib import Path
from dataclasses import dataclass
import frontmatter

# Defining the structure for each document
@dataclass
class Document:
    # REVIEW [readability]: stray empty comment marker left on the line below -
    # either finish the thought or delete it. Small thing, but a marker reads
    # this as "unfinished draft" and it costs you nothing to catch.
    path: str #
    name: str
    content: str
    metadata: dict
    # REVIEW [scalability]: no `modified` / content-hash field. Because you don't
    # record it, `index_chunks` downstream has no way to tell which documents
    # actually changed, so every run re-embeds the entire corpus (see the note in
    # embedding.py). Capturing `md_path.stat().st_mtime` here - one line - is what
    # unlocks incremental indexing later. Design decisions in the loader propagate
    # all the way to your API bill.

def load_vault(vault_dir:str | Path) -> list[Document]:
    """Loads the full corpus in and ingests it as a Document dataclass to include path, file name, file content & frontmatter metadata"""
    # REVIEW [readability]: this docstring is one 130-character line. Wrap it, and
    # split it into a one-line summary followed by an Args/Returns block. Compare
    # it with your `search()` docstring in querying.py, which is laid out properly -
    # be consistent across the codebase.

    vault = Path(vault_dir)
    # REVIEW [logic]: no check that `vault` exists or is a directory. Point this at
    # a typo'd path and `rglob` cheerfully yields nothing, so you get an empty
    # corpus, an empty index, and a chatbot that answers "I don't know" to
    # everything - with no error anywhere to tell you why. Fail loudly:
    #   if not vault.is_dir(): raise NotADirectoryError(vault)
    # Rule of thumb: silent empty results are worse than exceptions.

    documents = []
    for md_path in sorted(vault.rglob('*.md')):

        # Using pathlib to open .md files and extract the relevant YAML frontmatter
        with md_path.open(encoding='utf-8') as f:
            post = frontmatter.load(f)
        # REVIEW [logic]: one file with broken YAML frontmatter raises here and
        # aborts the entire load - 30 good notes discarded because of 1 bad one.
        # For a batch ingestion job, prefer to collect-and-continue:
        #   try: post = frontmatter.load(f)
        #   except Exception as e: logger.warning("skipping %s: %s", md_path, e); continue
        # Then report the skip count at the end. Partial success beats total failure.

        content = post.content
        metadata = post.metadata
        path = str(md_path.relative_to(vault)).replace("\\",'/') # Ensures clean path regardless of operating system
        # REVIEW [readability]: `md_path.relative_to(vault).as_posix()` is the
        # stdlib's own way to say this and removes the manual separator swap.
        # Reach for the library method before hand-rolling string surgery.
        file_name = str(metadata.get('title') or md_path.stem)
        # REVIEW [logic]: good use of `or` to fall back when `title:` is absent OR
        # blank. Worth a brief comment saying that's deliberate - a later reader
        # may "tidy" it into `metadata.get('title', md_path.stem)`, which quietly
        # behaves differently when the key exists but is empty.

        documents.append(
            Document(path = path,
                     name = file_name,
                     content=content,
                     metadata=metadata)
        )
        # REVIEW [readability]: inconsistent spacing around `=` within a single
        # call - `path = path` then `content=content`. PEP 8 wants no spaces for
        # keyword arguments. Run `ruff format` over the project and stop spending
        # attention on this class of thing entirely.
    return documents
    # REVIEW [logic]: empty-corpus case again - returning `[]` here is treated as
    # success by every caller. `app.py` will happily report "Indexed 0 chunks".
    # Decide who owns that check and make it explicit.
