"""Turning file system text into vectors using OpenAI's embedding model"""
# =============================================================================
# MARKER'S NOTE - embedding.py
# Short and readable, and the deterministic-ID scheme is the right instinct -
# it makes re-indexing idempotent instead of duplicating the corpus on every
# run. Well judged. The marks lost here are all about *cost and volume*: this
# module makes exactly one API call for the entire corpus, re-embeds every
# chunk whether or not it changed, and has no retry path. All three are
# invisible at 25 notes and fatal at 2,500.
# =============================================================================
import chromadb
from openai import OpenAI
from rag_build.config import EMBEDDING_MODEL, _find_root
# REVIEW [readability]: you are importing `_find_root`, a deliberately private
# helper (leading underscore = "not part of this module's public API"), across a
# module boundary. Either it is shared - in which case drop the underscore and
# expose it properly - or config.py should export the already-computed
# `PROJECT_ROOT` constant, which is what you actually want here. As written the
# underscore is telling the reader a lie.

PERSIST_DIR = _find_root() / ".chroma"
# REVIEW [redundancy]: config.py already computed this as `PROJECT_ROOT`. You are
# re-walking the filesystem on import to recalculate a value you were handed.
# `from rag_build.config import PROJECT_ROOT` and use it.
# REVIEW [scalability]: the store location is hardcoded. The moment you want a
# separate index for tests (and you should - see below), you need this to be
# configurable. It belongs in config.py alongside your other tunables.
COLLECTION_NAME = "vault_chunks"

_client = OpenAI()
# REVIEW [redundancy]: `_client = OpenAI()` is constructed at import time in THREE
# separate modules - here, querying.py and response.py - plus a fourth throwaway
# inside `utils.check_openai`. That is the same object built four times, with
# four places to change if you ever need a timeout, a retry policy, or a proxy.
# Build it once in a shared module and import it. This is the clearest instance
# of copy-paste in the project.
# REVIEW [logic]: constructing the client at import time means merely IMPORTING
# this module reaches for credentials. Combined with the import-time key check in
# config.py, that makes the whole package un-importable without a valid .env -
# so you cannot unit-test the pure chunking logic, which needs no API at all.
# Prefer a lazily-created, cached client:
#   @lru_cache
#   def _get_client() -> OpenAI: return OpenAI()


def _flatten_metadata_lists(metadata:dict) -> dict:
    """Takes metadata dictionary and flatten any values that are lists to string values to pass to chromadb"""
    flat_metadata = {}

    for key,value in metadata.items():
        flat_metadata[key] = ','.join(value) if isinstance(value, list) else value

    return flat_metadata
    # REVIEW [logic]: this is the encode half of the lossy round-trip flagged in
    # chunking.py. Joining on "," is only reversible if no element contains a
    # comma - and your headings routinely will ("Bias, Variance..."). The decode
    # half lives in utils.py and response.py, which both `.split(',')`. Three
    # separate places now depend on an assumption that is not true. Use
    # `json.dumps(value)` / `json.loads(...)`, and put BOTH halves next to each
    # other in one module so they cannot drift apart.
    # REVIEW [logic]: `','.join(value)` raises TypeError on a list of non-strings -
    # e.g. YAML frontmatter `tags: [1, 2]` parses to ints. Use
    # `','.join(map(str, value))`, or validate frontmatter on load.
    # REVIEW [logic]: Chroma also rejects `None` values in metadata. A frontmatter
    # key present but blank (`title:`) parses to None and will error at upsert
    # time - a long way from the note that caused it. Filter Nones out here.
    # REVIEW [readability]: the body is a dict comprehension:
    #   return {k: ','.join(v) if isinstance(v, list) else v for k, v in metadata.items()}

def embed_texts(texts:list[str]) -> list[list[float]]:
    """Embed many chunks of text in one API call. Returns one vector per input"""
    # REVIEW [scalability] IMPORTANT: "one API call" is precisely the problem. The
    # OpenAI embeddings endpoint caps a single request at 2,048 inputs and
    # ~300,000 tokens. Your corpus is under that today, so this works; cross
    # either limit and the whole indexing run fails with a 400 rather than
    # degrading. This function should batch internally:
    #   for i in range(0, len(texts), BATCH): ...embed slice, extend results...
    # It is ~4 lines and it is the difference between a demo and a pipeline.
    # REVIEW [scalability]: no retry/backoff. Embedding calls hit 429 rate limits
    # and transient 5xx routinely; one blip currently loses the entire run. The
    # OpenAI SDK takes `max_retries=` on the client constructor - set it once at
    # the shared client and you have covered every call site in the project.

    response = _client.embeddings.create(model = EMBEDDING_MODEL,input=texts)

    return [item.embedding for item in response.data]
    # REVIEW [logic]: you rely on `response.data` coming back in request order.
    # That is true, but each item carries an explicit `.index` field precisely so
    # you don't have to rely on it. Sorting by `item.index` costs nothing and
    # removes a silent-corruption failure mode where every chunk gets the wrong
    # vector - a bug that produces no error, just quietly terrible retrieval.

def get_collection():
    """calls the chromadb collection"""
    # REVIEW [readability]: missing return type annotation - `-> chromadb.Collection`.
    # Every other function in this module is annotated; this one breaks the
    # pattern, and it is the one whose return type is least guessable.
    # REVIEW [readability]: the docstring restates the function name. Say what a
    # reader cannot infer: that it creates the collection if absent, and that the
    # distance metric is fixed to cosine.
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    # REVIEW [efficiency]: a fresh client is constructed on EVERY call, and this
    # is called on every single search (via querying.search). Chroma caches
    # underlying instances internally so it is not as costly as it looks, but you
    # should not be depending on someone else's caching for your own hot path.
    # Decorate with `@lru_cache` and the collection handle is built once.
    # REVIEW [readability]: `path=` is typed as `str` in Chroma's signature and
    # you are passing a `Path`. It works, but you are outside the documented
    # contract - pass `str(PERSIST_DIR)` and be explicit.

    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={'hnsw:space':"cosine"}
        # REVIEW [logic]: `hnsw:space` is only applied when the collection is
        # first CREATED. If a collection already exists on disk with a different
        # metric, this argument is ignored silently and your `max_distance=0.8`
        # threshold in querying.py - which is calibrated for cosine distance -
        # becomes meaningless. Worth a comment here, or an assertion on the
        # returned collection's metadata.
    )

def index_chunks(chunks:list[str]) -> None:
    """upsert the chunks to chromadb, using document path and the chunk index within the document"""
    # REVIEW [readability] BUG: the type hint is WRONG - this takes `list[Chunk]`,
    # not `list[str]`. You then access `chunk.metadata` and `chunk.text` on each
    # element, which no `str` has. A type checker would have caught this
    # instantly; it is a good argument for running `mypy` or `pyright` over the
    # project. An incorrect annotation is worse than none, because readers trust it.
    collection = get_collection()

    # Building deterministic ids for chromadb
    ids = [f'{chunk.metadata['source']}_{chunk.metadata['index']}' for chunk in chunks]
    # REVIEW [logic]: deterministic IDs + `upsert` = idempotent re-indexing. This
    # is the right design and deserves credit - re-running the app updates rows
    # rather than duplicating the corpus.
    # REVIEW [logic]: but note what it does NOT do - if you DELETE or rename a
    # note, its old chunks stay in the store forever and keep being retrieved.
    # A full rebuild needs to reconcile: fetch existing IDs, and delete any no
    # longer present in `ids`. Currently the index only ever grows.
    # REVIEW [readability]: nested same-type quotes inside an f-string
    # (`f'{chunk.metadata['source']}'`) is legal only on Python 3.12+. Your
    # `requires-python = ">=3.13"` makes it valid, but it is hard to read and many
    # editors still mis-highlight it. Prefer `f"{chunk.metadata['source']}"`.

    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(chunk_texts)
    metadata = [_flatten_metadata_lists(chunk.metadata) for chunk in chunks]
    # REVIEW [efficiency] IMPORTANT: every call re-embeds every chunk, including
    # the ones that have not changed since the last run. Embeddings are the one
    # part of this pipeline you pay real money for, and they are perfectly
    # cacheable - the same text always yields the same vector. Hash each chunk's
    # text, compare against what is already stored, and embed only the delta.
    # With `@st.cache_resource` in app.py you avoid this within one session, but
    # every restart, redeploy or cache eviction currently re-buys the whole corpus.
    # REVIEW [readability]: three separate list comprehensions each walking
    # `chunks`. Not a performance concern at this size, but one loop building all
    # three lists reads more directly and keeps the three lists provably aligned -
    # right now their correspondence is positional and entirely implicit.

    collection.upsert(
        ids         = ids,
        embeddings  = embeddings,
        documents   = chunk_texts,
        metadatas    = metadata
        # REVIEW [readability]: the alignment padding slips by one space here, and
        # the variable is singular (`metadata`) while the parameter is plural
        # (`metadatas`) - it holds a list, so name it `metadatas`. Aligned-equals
        # formatting like this is also exactly what a formatter will undo, so you
        # are creating churn for yourself; let the tool decide and move on.
    )
    # REVIEW [logic]: no verification that the upsert landed. A cheap
    # `collection.count()` afterwards, logged, turns "did indexing work?" from a
    # guess into a fact - and would immediately surface an empty-corpus run.
