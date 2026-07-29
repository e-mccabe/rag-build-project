# =============================================================================
# MARKER'S NOTE - app.py
# Idiomatic Streamlit - the session_state message pattern and the walrus in
# `if prompt := st.chat_input(...)` are both exactly how this should be written,
# and caching the index build was the right instinct. Three things hold it back:
# no module docstring, no error handling anywhere (every exception raised
# in the modules below becomes a red traceback in the user's browser), and the
# chat history is stored but never actually used, so this is not yet a
# conversational app - it is a sequence of unrelated one-shot questions.
# =============================================================================
import streamlit as st

from rag_build.loading import load_vault
from rag_build.chunking import chunk_all_documents
from rag_build.embedding import index_chunks
from rag_build.response import retrieval, generate_stream
# REVIEW [readability]: the UI layer imports four separate internals of the
# pipeline and wires them together itself. That means app.py has to KNOW the
# pipeline's shape - load, then chunk, then index - so any change to that
# sequence forces a UI change. Expose one `build_index()` (or an `Indexer`) from
# the package and let the app call that. UIs should depend on a small surface.


st.set_page_config(page_title='Second Brain Assistant',page_icon='~',initial_sidebar_state='expanded')
# REVIEW [readability]: `page_icon='~'` renders as a literal tilde in the browser
# tab. Presumably a placeholder - Streamlit takes an emoji here (e.g. '🧠').
st.title('Ask the second brain')

st.sidebar.title('Settings')
st.sidebar.write('Navigation Page')
# REVIEW [redundancy]: a sidebar titled "Settings" containing the text
# "Navigation Page" and no settings and no navigation. Placeholder UI is worse
# than no UI - it advertises functionality that does not exist. Either wire up
# real controls (top_k, max_distance and model choice are the obvious candidates,
# and exposing them would make the app genuinely useful for tuning) or remove it.

# Building the vector store upon loading and caching it
@st.cache_resource
def build_index()-> int:
    # REVIEW [readability]: `@st.cache_resource` is for unserialisable singletons
    # (connections, models); this function returns an `int`, which is
    # `@st.cache_data` territory. You are really caching a SIDE EFFECT - "has
    # indexing run?" - and returning a count incidentally. That works, but say so
    # in a comment, because the choice looks like a mistake to a Streamlit reader.
    # REVIEW [logic]: no docstring on the one function in this file.

    documents = load_vault('data')
    # REVIEW [logic] BUG: the corpus path is hardcoded as the RELATIVE string
    # 'data', so the app only works when launched from the project root - run it
    # from anywhere else and you silently get an empty corpus (`rglob` on a
    # non-existent directory yields nothing, no error). config.py already defines
    # `PATHS.corpus` as an absolute path for exactly this purpose, and it is
    # currently unused anywhere in the project. Use it.
    chunks = chunk_all_documents(documents)
    index_chunks(chunks)
    # REVIEW [efficiency] IMPORTANT: this re-embeds the ENTIRE corpus on every
    # cold start - every redeploy, every cache eviction, every server restart.
    # `@st.cache_resource` only protects you within a running process. Since the
    # Chroma store is persistent on disk, the index usually already exists: check
    # `collection.count()` first and skip the work (or embed only changed chunks -
    # see the hashing note in embedding.py). As written, deploying this publicly
    # means paying the full embedding cost repeatedly for no benefit.
    # REVIEW [logic]: no try/except. If the OpenAI key is missing or the API is
    # down, the user gets a raw traceback rather than "The assistant is
    # unavailable, please check configuration". Streamlit's `st.error` plus
    # `st.stop()` is the idiomatic handling.
    return len(chunks)

n_chunks = build_index()
st.caption(f'Indexed {n_chunks} from the vault.')
# REVIEW [readability] BUG: missing noun - "Indexed 412 from the vault" should be
# "Indexed 412 chunks from the vault". User-facing copy deserves the same
# proofreading as code.
# REVIEW [logic]: `0` displays as a perfectly normal caption. An empty index is a
# broken app, so it should be an `st.error`, not a cheerful count.


if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
# REVIEW [readability]: correct, idiomatic Streamlit chat replay. Good.
# REVIEW [logic]: history is stored and REPLAYED but never SENT to the model -
# `generate_stream` receives only the current question (see response.py). So the
# transcript on screen implies a conversation the model has no knowledge of, and
# any follow-up like "why does that matter?" is answered blind. This is the
# biggest functional gap in the app. Either pass prior turns as messages, or
# rewrite each follow-up into a standalone query before retrieving (the latter is
# usually better for RAG, since pronouns retrieve badly).
# REVIEW [scalability]: unbounded growth - a long session eventually exceeds the
# context window once history IS passed. Cap it to the last N turns.


if prompt := st.chat_input('Ask something'):
    # REVIEW [readability]: nice use of the walrus operator - concise and clear.
    # REVIEW [readability]: `prompt` here means "the user's question", but
    # everywhere else in the project `prompt` means "the assembled LLM input".
    # One word, two meanings, in code that sits either side of a module boundary.
    # Call this one `question`, matching the parameter name it is passed to.
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):

        hits = retrieval(prompt)
        # REVIEW [logic]: no spinner. Retrieval is an embedding call plus a full
        # LLM rerank - easily 2-4 seconds of the UI appearing frozen before the
        # stream begins. Wrap it in `with st.spinner('Searching notes...')`.
        # REVIEW [logic]: unguarded. `_inspect_collection` raises ValueError on an
        # empty collection and the OpenAI calls can raise anything; all of it lands
        # in the browser as a traceback. This is the one place in the project where
        # a try/except is not optional, because it is the boundary with the user.
        if not hits:
            answer = "I don't have anything in the corpus about that."
            # REVIEW [redundancy]: a THIRD wording of the same refusal - `ask()` in
            # response.py says "No Answer Found in Corpus" and SYSTEM_PROMPT
            # instructs "I cannot find the answer to your questions in the provided
            # context". One message, defined once in config.py.
            st.markdown(answer)
            sources = []
            # REVIEW [redundancy] BUG: `sources` is assigned and NEVER READ -
            # not in this branch, not below, nowhere in the file. Dead code, and it
            # is a fossil of a feature you have half-built: `ask()` computes real
            # source breadcrumbs, but this UI uses `generate_stream()`, which
            # cannot return them (see response.py). So citations - the whole point
            # of the "[ID: n, Source: ...]" rules in your SYSTEM_PROMPT - are never
            # displayed. Fixing the streaming path to also yield sources would let
            # you render them under each answer, which is the single most visible
            # improvement available in this file.

        else:
            answer = st.write_stream(generate_stream(prompt,hits))
            # REVIEW [logic]: this works only because `st.write_stream` knows how to
            # unwrap raw OpenAI chunk objects - your generator yields SDK objects
            # rather than text. It ties a library module to this specific UI. See
            # the note in response.py.

    st.session_state.messages.append({'role':'assistant','content':answer})
    # REVIEW [logic]: correct that this sits outside the `if/else` so both paths
    # are recorded - a common place to introduce a bug, and you avoided it.

#        st.markdown(response)
# REVIEW [redundancy]: commented-out dead code referencing a variable
# (`response`) that does not exist in this file. Delete it.
