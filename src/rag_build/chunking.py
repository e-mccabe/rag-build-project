"""Splitting corupus documents into small sections with sufficient content and semantic meaning"""
# =============================================================================
# MARKER'S NOTE - chunking.py
# The heading-stack algorithm is genuinely good work - maintaining ancestor
# breadcrumbs with a single `del` slice is elegant and I'd be pleased to see it
# in a submission. But this module contains the most consequential weakness in
# the project: the module docstring promises chunks "with sufficient content",
# and nothing here measures content at all. There is no size floor, no size
# ceiling, and the heading text itself is discarded from the embedded string.
# For a retrieval system, chunking quality sets the ceiling on every metric you
# will ever measure downstream. Fix this before you tune anything else.
# Typo in the docstring above: "corupus" -> "corpus".
# =============================================================================

from dataclasses import dataclass
from rag_build.loading import Document

import re
# REVIEW [readability]: import ordering. PEP 8 asks for stdlib, then third-party,
# then local - each in its own block. Here `re` (stdlib) sits below a local
# import. Same slip in config.py. A formatter/linter settles this permanently.

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
# REVIEW [logic]: this matches `#` inside fenced code blocks too. A Python
# comment such as `# Fit the model` at the start of a line inside a ```python
# fence is read as an H1 and splits your document there. In an ML notes corpus
# full of code samples this will genuinely fire. Track whether you are inside a
# ``` fence while scanning and skip heading detection when you are.
# REVIEW [logic]: `{1,4}` silently ignores H5/H6 - they fall through to the
# `else` branch and get absorbed as body text. Deliberate? If so, say so here;
# an undocumented magic bound is a question mark for every future reader.

# Define data structure for each chunk
@dataclass
class Chunk:
    text: str
    metadata : dict
    # REVIEW [readability]: space before the colon on `metadata`, none on `text`.
    # Be consistent.


def _split_string_by_headers(content:str) -> list[tuple[list[str],str]]:
    """Using the # characters in .md headings to split the document into distinct sections"""
    # REVIEW [readability]: the return type `list[tuple[list[str],str]]` is doing
    # a lot of unexplained work. Which element is the breadcrumb, which is the
    # body? A `NamedTuple` (or documenting it as "(heading_path, body_text)")
    # turns `section[0]` into `section.headings` at the call site. Naming things
    # is the cheapest documentation you will ever write.

    sections: list[tuple[list[str],str]] = []
    heading_stack: list[str] = []
    compiled_text: list[str] = []


    for line in content.splitlines():

        heading_match = HEADING_RE.match(line)
        if heading_match:
            # Select text after the has signs
            # REVIEW [readability]: typo - "has signs" -> "hash signs".
            heading_text = heading_match.group(2).strip()
            # If a heading is hit then take all the take in compiled text as the previous section
            # REVIEW [readability]: "all the take in" - garbled. Comments that
            # have decayed like this are worse than no comment, because the
            # reader stops to decode them and gains nothing.
            text = "\n".join(compiled_text).strip()

            if text:
                sections.append((list(heading_stack),text))
                # Clear all the text so it is clean for the next section
                compiled_text.clear()
            # REVIEW [logic] BUG: `compiled_text.clear()` is INSIDE `if text:`.
            # When a section body is empty or whitespace-only - which is exactly
            # what happens with consecutive headings, e.g. `# Title` immediately
            # followed by `## Subtitle` - the buffer is never cleared, and those
            # blank lines leak forward into the next section's text. Harmless
            # today because `.strip()` removes them, but the invariant "the
            # buffer is empty after a heading" is violated, and the next person
            # to extend this loop will be bitten. Move `.clear()` outside the
            # `if`, so it runs on every heading regardless.
            # REVIEW [logic]: good catch using `list(heading_stack)` to snapshot -
            # without the copy every section would alias the same mutating list.
            # Note that the tail append on line ~52 does NOT make that copy.

            # Calculate the heading level based on the number of # Characters
            heading_level = len(heading_match.group(1))
            # Delete any remaining n level headings or deeper headings from the previous section
            del heading_stack[heading_level - 1:]
            # The vacated slot at index n-1 now holds this heading, ancestors above it stay in scope
            heading_stack.append(heading_text)
            # REVIEW [logic]: this is the strongest idea in the module and the
            # three explanatory comments are exactly right - keep them. One edge
            # case to consider: a document that opens at `###` with no `#` above
            # it produces a stack with gaps collapsed, so the breadcrumb silently
            # implies a hierarchy that isn't in the file. Worth a test.

        else:
            compiled_text.append(line)

    # Compile anything remaining in compiled text to a full piece of text
    text = "\n".join(compiled_text).strip()
    if text or heading_stack:
        sections.append((heading_stack,text))
        # REVIEW [logic] BUG: inconsistency with the loop body above - here you
        # append `heading_stack` itself, not `list(heading_stack)`. The final
        # section therefore holds a live reference to the list. Nothing mutates
        # it after this point *today*, so the bug is latent rather than active,
        # but the asymmetry with line ~35 is the tell. Copy it here too. When two
        # nearly identical lines differ, one of them is a mistake.
        # REVIEW [logic]: the `or heading_stack` condition emits a section with
        # `text == ''` for a document ending on a bare heading. That empty chunk
        # is then embedded downstream - you pay for an API call and insert a
        # meaningless vector that can still be returned as a search hit. Require
        # non-empty text before emitting.

    return sections


def chunking_document(document:Document) -> list[Chunk]:
    """Building and indexing the chunk dataclass including the content, metadata and breadcrumb to the section"""
    chunks: list[Chunk] = []
    document_chunk_index: int  = 0
    # REVIEW [readability]: a hand-rolled counter incremented at the bottom of
    # the loop is what `enumerate()` exists to remove. Use
    # `for i, (headings, section) in enumerate(...)` and delete both this line
    # and the `+= 1` below. Manual counters are a classic source of off-by-one
    # bugs when someone later adds a `continue`.

    for headings, section in _split_string_by_headers(document.content):

        chunks.append(
            Chunk(
                text = f"\n{section}\n",
                # REVIEW [logic] IMPORTANT: the heading text is thrown away here.
                # The breadcrumb goes into metadata, but the string you actually
                # EMBED is the body only. So the chunk under "## K-Means
                # Clustering" may never contain the words "K-Means" or
                # "clustering", and a user asking "how does K-Means work?" cannot
                # match it on semantics. This is the single highest-value fix in
                # the project: prepend the breadcrumb to the embedded text, e.g.
                #   text = f"{' > '.join(headings)}\n\n{section}"
                # You already build exactly this breadcrumb string in
                # utils.generate_numbered_context_strings - you're computing the
                # right thing, just not at embedding time.
                # REVIEW [efficiency]: the leading/trailing "\n" padding serves no
                # purpose - it is tokenised, embedded and paid for on every chunk.
                # Strip it.
                metadata={
                    "source":document.path,
                    "file":document.name,
                    "headings":headings,
                    # REVIEW [logic]: `headings` is a list, and Chroma cannot
                    # store lists - so `_flatten_metadata_lists` in embedding.py
                    # joins it with "," and `utils` later splits it back on ",".
                    # That round-trip is LOSSY: any heading containing a comma
                    # (e.g. "Bias, Variance and the Trade-off") is silently split
                    # into two fake breadcrumb levels. Use a separator that
                    # cannot occur in a heading, or `json.dumps` the list. Encode
                    # and decode must be exact inverses of each other.
                    "index":document_chunk_index,
                    **document.metadata
                    # REVIEW [logic]: unpacking user-controlled frontmatter LAST
                    # means a note whose YAML contains `source:` or `index:`
                    # overwrites your own keys and corrupts the chunk's identity -
                    # which is what `index_chunks` builds its primary key from.
                    # Put `**document.metadata` FIRST so your controlled fields
                    # win, or namespace the frontmatter under a `fm_` prefix.
                    # Never let external data overwrite your invariants.
                }
                )
                    )
                    # REVIEW [readability]: that closing-bracket ladder is
                    # misaligned and hard to scan. A formatter fixes it instantly.
        document_chunk_index += 1

    return chunks
    # REVIEW [logic] IMPORTANT: nowhere in this function is chunk SIZE considered.
    # Two consequences, both material for a RAG system:
    #  1. Too large - a long section under one heading becomes one enormous chunk.
    #     `text-embedding-3-small` truncates at 8191 tokens, so the tail of a long
    #     note is silently never indexed. You will not get an error; you will just
    #     never retrieve it.
    #  2. Too small - a heading with a single line under it becomes its own chunk,
    #     producing a near-meaningless vector that adds noise to every search.
    # The standard remedy is a token budget: merge sections below a floor (~100
    # tokens) into their parent, and split sections above a ceiling (~800 tokens)
    # on paragraph boundaries with a small overlap. `tiktoken` is already in your
    # pyproject dependencies but never imported - this is presumably what you
    # intended it for, and your README's "Elements to be added / Large Paragraphs
    # logic" confirms you have already spotted it. Prioritise it.

def chunk_all_documents(documents:list[Document]) -> list[Chunk]:
    """Full chunking of the whole corpus of .md documents"""
    chunked_documents: list[Chunk] = []

    for document in documents:
        chunked_documents.extend(chunking_document(document))

    return chunked_documents
    # REVIEW [readability]: this whole body is one comprehension:
    #   return [c for doc in documents for c in chunking_document(doc)]
    # Correct as written - just more ceremony than the job needs.
    # REVIEW [scalability]: everything is held in memory - full corpus, then full
    # chunk list, then (in embedding.py) the full embedding matrix. Fine for ~25
    # notes; at a few thousand you would want this to be a generator so the
    # pipeline streams rather than accumulates. Worth knowing where the ceiling
    # is even when you are comfortably under it.
