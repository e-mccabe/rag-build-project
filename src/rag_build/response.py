"""Generate the response output from a users query"""
# =============================================================================
# MARKER'S NOTE - response.py
# Short and mostly clear, but this is where the project's worst duplication
# lives: `ask` and `generate_stream` are the same function written twice, and
# they have already begun to diverge (one has a no-hits guard, the other does
# not; one returns sources, the other silently cannot). That divergence is the
# textbook consequence of copy-paste, and it happened within 20 lines. Extract
# the shared part. Also note "a users query" in the docstring needs an
# apostrophe.
# =============================================================================
from openai import OpenAI

from rag_build.querying import search,rerank
from rag_build.utils import generate_numbered_context_strings
from rag_build.config import RESPONSE_MODEL,SYSTEM_PROMPT
# REVIEW [readability]: missing space after commas in all three import lines.
# Trivial in isolation, but it is the same slip on every line of the file - a
# formatter removes this entire category of comment from your next review.

_client = OpenAI()
# REVIEW [redundancy]: fourth construction of the OpenAI client in this project
# (embedding.py, querying.py, here, and utils.check_openai). One shared, lazily
# created client would let you set timeouts and retries in a single place.

def retrieval(question:str, **search_kwargs) -> list[dict]:
    """Based of user query generate the relevant chunks for the vector database"""
    # REVIEW [readability]: "Based of" -> "Based on", and "for the vector
    # database" -> "from the vector store". Also state what it returns: hits
    # already reranked, at most `top_n`.

    hits = search(query=question,**search_kwargs)

    if not hits:
        return []
    # REVIEW [redundancy]: `rerank` already opens with `if not hits: return hits`,
    # so this guard is doing the same job a second time. Not harmful, but two
    # layers of the same check means neither function clearly owns the contract.
    # Pick one - I would keep it in `rerank` (defend at the boundary of the
    # function that would otherwise crash) and delete it here.

    return rerank(question,hits)
    # REVIEW [scalability]: `**search_kwargs` forwards to `search` only, so
    # `rerank`'s `top_n` cannot be reached from any caller - not from app.py, not
    # from evaluate.py. That means your evaluation harness cannot sweep the single
    # most obvious retrieval parameter. Add an explicit `top_n` parameter here and
    # pass it through. Prefer explicit, forwarded parameters over `**kwargs` when
    # the set of options is knowable - `**kwargs` also destroys autocomplete and
    # lets typos like `top_kk=20` pass silently to a function that will reject them
    # only at runtime.

def ask(question: str,**search_kwargs) -> dict:
    # REVIEW [readability]: no docstring at all, on a public function that is the
    # main programmatic entry point to the whole system and the one evaluate.py
    # depends on. Every other function in this file has one. At minimum document
    # the returned dict's keys - `answer`, `sources`, `ids` - because callers
    # currently have to read the body to discover them.
    # REVIEW [readability]: `-> dict` is unspecific. A small `Answer` dataclass or
    # `TypedDict` would let evaluate.py's `response['ids']` be checked statically
    # instead of failing at runtime on a typo.

    hits = retrieval(question,**search_kwargs)

    if not hits:
        return {
            'answer':'No Answer Found in Corpus',
            'sources':['N/A'],
            'ids':['N/A']
        }
    # REVIEW [logic]: good that the empty case is handled and that the shape of
    # the returned dict stays consistent - callers don't need a special path.
    # REVIEW [logic] BUG: but the SENTINEL VALUES are wrong. `ids: ['N/A']` is a
    # list of length 1 containing a fake id. Downstream in evaluate.py,
    # `case_recall` computes `len(retrieved) = 1` and divides by it - so a
    # question the system completely failed to retrieve for is scored with a
    # non-zero denominator instead of being recognised as an empty retrieval.
    # Use empty lists (`[]`), which are falsy and which your metric code already
    # guards against. Sentinel strings inside typed collections are a reliable
    # source of downstream arithmetic bugs.
    # REVIEW [readability]: this fallback string differs from the one hardcoded in
    # app.py ("I don't have anything in the corpus about that.") and from the
    # refusal wording in SYSTEM_PROMPT. Three different ways of saying "I don't
    # know" - a user could see any of them. Define it once in config.py.

    context_string = generate_numbered_context_strings(hits)

    prompt =  f'User Question: {question}. **Retrieved Context: {context_string}'
    # REVIEW [redundancy]: identical to the line in `generate_stream` below and to
    # the one in querying.rerank. Third copy. Extract `build_prompt(question, hits)`.
    # REVIEW [readability]: double space after `=`, and the same unmatched `**`.


    response = _client.chat.completions.create(
        model=RESPONSE_MODEL,
        max_tokens=500,
        # REVIEW [logic]: 500 tokens is a hard ceiling on the answer. Your
        # SYSTEM_PROMPT asks for a conclusion, supporting arguments, underlying
        # data AND inline citations - that structure will hit 500 tokens and be
        # truncated mid-sentence, with no error and no indication to the user.
        # Either raise the budget or ask for a shorter format; as it stands the
        # prompt and the token budget are pulling against each other.
        # REVIEW [logic]: no `temperature`. For a strictly-grounded citation task
        # you want `temperature=0` - the default sampling actively works against
        # the "never make up information" instruction in your system prompt.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        )
    # REVIEW [logic]: no error handling around a network call. A rate limit or
    # timeout propagates as a raw exception - into a red traceback in the browser
    # (app.py) or an aborted evaluation run partway through (evaluate.py).
    # REVIEW [readability]: single-quoted keys elsewhere in the project,
    # double-quoted here. Harmless, but pick one.
    # REVIEW [logic]: conversation history is never sent. `st.session_state.messages`
    # in app.py accumulates the whole chat and none of it reaches the model, so
    # every follow-up ("what about its assumptions?") is answered with no idea
    # what "it" refers to. For a chat interface this is a functional gap, not a
    # polish item - either pass prior turns, or rewrite the follow-up into a
    # standalone query before retrieving.

    sources = [f'{hit['metadata']['source']}/{'/'.join(hit['metadata']['headings'].split(','))}' for hit in hits]
    # REVIEW [readability]: this is a lot of work for one line - two f-strings,
    # nested same-type quotes, a split and a join. Give it a loop or a small
    # helper; you will thank yourself when it needs changing.
    # REVIEW [logic]: `.split(',')` is the decode half of the lossy round-trip
    # flagged in chunking.py and embedding.py. This is now the THIRD site
    # depending on "no heading contains a comma". When one assumption is
    # replicated across three modules, it is guaranteed to be violated eventually
    # and painful to fix when it is.
    # REVIEW [logic] BUG: `hit['metadata']['headings']` raises KeyError for any
    # chunk indexed before `headings` existed, and raises AttributeError if the
    # value is not a string. `.get('headings', '')` costs nothing.
    ids = [hit['id'] for hit in hits]
    return {
        'answer':response.choices[0].message.content,
        'sources':sources,
        'ids':ids
    }
    # REVIEW [logic]: `message.content` is `None` when the model returns no text
    # (refusal, or a finish_reason of "length" with nothing emitted). Callers
    # assume a string. Coerce with `or ''`.

def generate_stream(question: str,hits:list[dict]):
    # REVIEW [readability]: no docstring and no return annotation on a public
    # function. It is a generator - `-> Iterator[...]` - and that is not obvious
    # from the name.
    # REVIEW [redundancy] IMPORTANT: this function is `ask` with `stream=True`.
    # The prompt construction, the model, the token budget and the message list
    # are all duplicated verbatim. This is the clearest refactor in the project:
    #     def _build_messages(question, hits) -> list[dict]: ...
    # then `ask` and `generate_stream` each become about four lines. Note the
    # divergence has ALREADY started - `ask` guards `if not hits`, this does not;
    # `ask` returns sources, this cannot. That is exactly how duplicated code
    # rots, and it took only twenty lines to begin.
    # REVIEW [logic]: the signature differs too - `ask` takes `**search_kwargs`
    # and retrieves internally, while this takes `hits` and expects the CALLER to
    # have retrieved. Two entry points with two different contracts for one
    # operation. app.py consequently has to know to call `retrieval` first, which
    # is why the no-hits branch had to be reimplemented up there in the UI.

    context_string = generate_numbered_context_strings(hits)

    prompt =  f'User Question: {question}. **Retrieved Context: {context_string}'

    stream = _client.chat.completions.create(
         model=RESPONSE_MODEL,
         # REVIEW [readability]: one extra leading space misaligns this argument
         # from the rest of the block.
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        stream=True
        )

    for piece in stream:
        yield piece
    # REVIEW [logic]: you are yielding raw SDK chunk OBJECTS, not text. This
    # happens to render because `st.write_stream` special-cases OpenAI chunks -
    # so a library module is now silently depending on the behaviour of your UI
    # framework. Yield the text and the coupling disappears:
    #     text = piece.choices[0].delta.content
    #     if text: yield text
    # That also makes the function testable and usable from a CLI or an API - as
    # written it only works inside Streamlit.
    # REVIEW [logic]: the streamed answer's `sources` and `ids` are never
    # returned, so app.py cannot display citations for streamed responses even
    # though `ask` computes them. That is a user-visible feature lost purely to
    # the API split above - more evidence the two paths should share one core.
    # REVIEW [readability]: `for piece in stream: yield piece` is `yield from stream`.
