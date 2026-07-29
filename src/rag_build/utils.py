# =============================================================================
# MARKER'S NOTE - utils.py
# Note first that this file has NO MODULE DOCSTRING - the only module in the
# project missing one, and you were otherwise consistent about them.
# Second, and more important: "utils" is not a real module name. It describes
# where code went when it had nowhere else to go, not what the code does. Here
# it holds one prompt-formatting function and one connectivity smoke test -
# two entirely unrelated jobs. The formatter belongs next to prompt building
# (see the duplicated prompt template flagged in response.py); the smoke test
# belongs in a scripts/ or tests/ directory. Grab-bag modules attract more
# junk over time and eventually become impossible to import without side
# effects. Name modules for what they contain.
# =============================================================================
from openai import OpenAI
from rag_build.config import EMBEDDING_MODEL, RESPONSE_MODEL

def generate_numbered_context_strings(hits:list[dict]) -> str:
    """
    Takes the resulting list of chunks & their meta and creates a single string to pass LLM
    
    The output string is in the form

    '<0 example_file: Heading 1 > Heading 2 >

    Example Text

    <1 next_file: Heading 1 > Heading 2 >

    Example Text'
    """
    # REVIEW [readability] BUG: the docstring EXAMPLE IS WRONG. It shows numbering
    # starting at `<0`, but the code below uses `enumerate(hits, 1)` and starts at
    # `<1`. That matters more than a typo normally would, because these numbers are
    # the identifiers the reranker scores against and that SYSTEM_PROMPT tells the
    # model to cite as "[ID: 1, ...]". A reader trusting this docstring would
    # write off-by-one code. Documentation that contradicts the code is worse than
    # none.
    # REVIEW [readability]: the example also omits the closing `>` shown in the
    # actual f-string. If a docstring is going to specify a format precisely, it
    # has to be precise.
    context_strings = []

    for i, hit in enumerate(hits,1):
        # REVIEW [logic]: 1-based numbering is the right call here - it matches how
        # the rerank prompt and the citation format both refer to passages. Worth
        # a one-line comment saying so, since the choice is deliberate.

        file = hit['metadata']['file']
        headings = hit['metadata']['headings'].split(',')
        # REVIEW [logic]: the third `.split(',')` decode site (chunking.py encodes,
        # embedding.py joins, response.py and this file both split). As flagged
        # elsewhere, this is lossy for any heading containing a comma. Fix the
        # encoding once, and put encode/decode in the same module so they cannot
        # drift.
        # REVIEW [logic]: unguarded `[...]` indexing on both keys - a chunk missing
        # `file` or `headings` raises KeyError deep inside prompt construction,
        # which is a confusing place to debug from. Use `.get(..., '')`.
        breadcrumb = f'<{i} {file}: {' > '.join(headings)}>'
        full_text = f'{breadcrumb}\n\n{hit['text']}'
        # REVIEW [readability]: nested same-type quotes inside f-strings again -
        # valid on 3.12+, but hard to read and poorly highlighted by many editors.
        # Alternate the quote style: f"{hit['text']}".
        # REVIEW [logic]: `hit['text']` is inserted with no boundary marker, so
        # chunk content runs directly into the next `<n ...>` breadcrumb. If a note
        # happens to contain a line like `<3 something:`, the model cannot tell
        # content from your delimiters. Use unambiguous delimiters (XML-style tags
        # are the conventional choice) rather than angle brackets that can occur
        # naturally in the text.
        context_strings.append(full_text)

    return '\n\n'.join(context_strings)
    # REVIEW [scalability]: no token budget. With `top_k=15` chunks of unbounded
    # size (see the chunking.py note - nothing caps section length), this string
    # can exceed the model's context window, and the API will reject the request
    # or silently truncate. Since this function builds the context for BOTH the
    # reranker and the answer generator, it is the natural place to enforce a
    # budget with `tiktoken` - which is already in your dependencies and not yet
    # imported anywhere.
    # REVIEW [redundancy]: the caller then wraps this in
    # `f'User Question: {question}. **Retrieved Context: {context_string}'` in
    # three separate places. Move that template in here as
    # `build_prompt(question, hits)` and all three copies collapse into one.

def check_openai() -> None:
    # REVIEW [readability]: no docstring. This is a diagnostic helper, so say what
    # "working" means and what it costs (two live API calls, billed).
    # REVIEW [logic]: nothing in the project calls this. Useful as a manual smoke
    # test, but with no `if __name__ == '__main__':` block there is no obvious way
    # to run it. Give it an entry point or move it to a scripts/ directory.

    client = OpenAI()
    # REVIEW [redundancy]: a fourth OpenAI client, constructed locally this time
    # while the other three are module-level. Inconsistent even in its inconsistency.
    embed = client.embeddings.create(model = EMBEDDING_MODEL, input= 'Hello World!')
    dims = len(embed.data[0].embedding)
    print(f'OpenAI working ({EMBEDDING_MODEL}) - test string embedded in {dims}-dimensional vector')
    # REVIEW [logic]: printing a success message but never checking anything means
    # this function cannot FAIL usefully - it either prints or raises a raw SDK
    # exception. For a health check, catch the exception and return a bool (or
    # raise a clear, wrapped error). As written it cannot be used programmatically,
    # only read by a human.
    # REVIEW [readability]: `print` is fine for a CLI diagnostic, but the rest of
    # the project also prints for real diagnostics (eval_generate.py, evaluate.py).
    # Standardise on the `logging` module: you get levels, timestamps and the
    # ability to silence it, none of which `print` offers.

    resp = client.responses.create(model = RESPONSE_MODEL,
                                   max_output_tokens= 20,
                                   input = 'Reply confirming model set up is successful')
    # REVIEW [readability]: this uses the Responses API (`client.responses.create`)
    # while every other call in the project uses Chat Completions
    # (`client.chat.completions.create`). So your "is my setup working?" check does
    # not actually exercise the code path the application uses - it could pass
    # while the real path fails. Test what you ship.

    print(f'Open AI ({RESPONSE_MODEL}) is working. Model response{resp.output_text}')
    # REVIEW [readability] BUG: missing separator - "Model response" runs straight
    # into the text. Should be `Model response: {resp.output_text}`. Also "Open AI"
    # is spelt "OpenAI" on the line above; be consistent within a single function.
