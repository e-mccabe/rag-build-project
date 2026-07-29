"""Configuration of project global variables and required keys"""
# =============================================================================
# MARKER'S NOTE - config.py
# Centralising models, paths and prompts in one module is the right decision and
# the "### ===== Section =====" banners make it easy to navigate. Good.
# Three things to work on. First, there is a live BUG: `Paths.eval_dir` is
# referenced as `PATHS.eval` by two other modules, and its value points at a
# directory that does not exist. Second, this module has import-time side
# effects that make the whole package un-importable without a valid .env, which
# is why none of your code can currently be unit-tested. Third, ~120 of these
# 169 lines are prompt text - see the note at SYSTEM_PROMPT.
# The prompts themselves are the best writing in the project. The Python around
# them is the part that needs attention.
# =============================================================================
import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass
# REVIEW [readability]: import ordering - `dotenv` is third-party and is sitting
# in the middle of your stdlib imports. PEP 8 wants stdlib, blank line,
# third-party. The same slip appears in four other modules; a linter fixes them
# all at once.

load_dotenv() # read .env file into the environment
# REVIEW [logic]: an import-time side effect. Importing ANY module in this
# package now reads the filesystem. It is the conventional place to call
# `load_dotenv()` so this is defensible - but note it silently does nothing if
# no .env exists, and combined with `_require` below the failure surfaces as a
# confusing error several lines later. Passing `override=False` explicitly would
# also document that real environment variables win over the file, which matters
# the moment you deploy this anywhere.

### ========= Helper Functions ========= 
def _require(name:str) -> str:
    """
    Runs a check whether if environment variables are available. 
    Fails with clear message if not available
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. {name} needs to be added to the .env file"
        )
    return value
    # REVIEW [readability]: the docstring reads "Runs a check whether if" - drop
    # one of "whether"/"if". Say what it RETURNS, too, since that is the part a
    # caller needs.
    # REVIEW [logic]: good - `if not value` catches the empty-string case as well
    # as the missing case, which `os.environ[name]` would not. Deliberate and
    # correct; worth a comment saying so.
    # REVIEW [readability]: the error message is genuinely helpful (names the
    # variable AND where to put it). This is what a good failure message looks
    # like - keep doing this.

def _find_root(marker: str = 'pyproject.toml') -> Path:
    """Identifies the project root from whatever directory it is run from"""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent/marker).exists():
            return parent
    raise FileNotFoundError(f'No file: {marker} found above {__file__}')
    # REVIEW [logic]: solid, portable approach - anchoring to `__file__` rather
    # than `Path.cwd()` means the project works regardless of where it is launched
    # from, and raising when the marker is absent is the right call. This is the
    # best-written function in the module.
    # REVIEW [readability]: it is marked private with `_`, yet embedding.py
    # imports it across a module boundary. Either make it public or export the
    # already-computed `PROJECT_ROOT` instead - see the note in embedding.py.
    # REVIEW [efficiency]: it walks the filesystem on every call, and embedding.py
    # calls it again at import to recompute a value this module already has.
    # `@lru_cache`, or just reuse `PROJECT_ROOT`.

### ========= Initialise Directories =========
PROJECT_ROOT = _find_root()

@dataclass(frozen=True)
class Paths:
    corpus  :Path = PROJECT_ROOT / 'data'
    eval_dir:Path = PROJECT_ROOT / 'eval_dir'
    # REVIEW [logic] BUG: TWO faults on this one line.
    # (1) NAME: eval_generate.py and evaluate.py both access `PATHS.eval`, which
    #     does not exist - the field is `eval_dir`. Both modules raise
    #     AttributeError immediately, so neither can run today.
    # (2) VALUE: it points at `PROJECT_ROOT/'eval_dir'`, but the actual directory
    #     holding evaluation_set.json - and the one listed in .gitignore - is
    #     `eval/`. So renaming the attribute alone is not enough; you would then
    #     read from an empty folder that gets created on demand. Both need fixing
    #     together, and settling on ONE name (`eval`) across config, callers and
    #     the filesystem is what prevents this recurring.
    # REVIEW [logic]: `frozen=True` is a good choice - config that cannot be
    # mutated at runtime. Credit for that.
    # REVIEW [readability]: `corpus` is defined here and used NOWHERE - app.py
    # hardcodes the string 'data' instead (see the note there). A config value
    # nothing reads is a config value that will drift out of sync with reality.
    # REVIEW [readability]: a frozen dataclass instantiated exactly once, with all
    # fields defaulted, is really just a namespace. That is fine and arguably
    # nicer than loose constants - but then it should be the ONLY way paths are
    # referenced, which brings us back to app.py's hardcoded 'data'.

PATHS = Paths()

### ========= API KEYS =========

OPENAI_API_KEY = _require("OPENAI_API_KEY")
# REVIEW [redundancy]: this module-level name is assigned and NEVER IMPORTED
# anywhere - the OpenAI SDK reads `OPENAI_API_KEY` from the environment itself.
# So the assignment is dead; only the SIDE EFFECT of `_require` (fail fast if the
# key is absent) actually matters. That intent is worth keeping, but say what you
# mean, so no one deletes it as unused:
#     _require("OPENAI_API_KEY")  # fail fast at import if the key is missing
# REVIEW [logic] IMPORTANT: because this runs at IMPORT time, and every module in
# the package imports config, NOTHING in this project can be imported without a
# valid .env - including the chunking and loading logic, which touch no API at
# all. That is precisely why there are no unit tests: the pure functions that are
# easiest to test cannot be imported in isolation. Moving the check into an
# explicit `validate_config()` called from app.py's entry point would keep the
# fail-fast behaviour while making the library importable and testable.

### ========= Project Model Choices =========
EMBEDDING_MODEL = 'text-embedding-3-small'

RESPONSE_MODEL = 'gpt-4o-mini'
# REVIEW [readability]: good - naming models once, here, is exactly what this file
# is for, and it is what let you use them consistently across five modules.
# REVIEW [scalability]: but the section is INCOMPLETE, and the gaps are the
# settings you most need to tune. `top_k=15`, `max_distance=0.8` and `top_n=5`
# are buried as function defaults in querying.py, and `max_tokens=500` is
# hardcoded at four separate call sites. Those are the retrieval system's real
# control panel; they belong here beside the model names. Until they do, your
# evaluate.py harness cannot sweep them, which means you cannot answer "is
# top_k=15 better than 25?" without editing source.
# REVIEW [logic]: one model constant is used for two very different jobs -
# reranking (cheap, structured, high volume) and answering (quality-critical).
# Splitting into `RERANK_MODEL` and `RESPONSE_MODEL` would let you use a small
# fast model for scoring and a stronger one for the final answer, which is the
# usual cost/quality trade-off in a RAG pipeline.
# REVIEW [logic]: nothing records the embedding dimension (1536 for this model).
# Changing `EMBEDDING_MODEL` silently invalidates every vector already in Chroma,
# and the mismatch surfaces as bad retrieval rather than as an error. Store the
# model name in the collection metadata and check it on load.

### ========= Project Prompts =========
# REVIEW [scalability] IMPORTANT (applies to all five prompts below): roughly 120
# of this file's 169 lines are prompt TEXT, not configuration. Consequences:
#  - Reading config.py to find a model name means scrolling past 100 lines of
#    English prose.
#  - Editing a prompt produces a diff against a .py file, so prompt changes and
#    code changes are tangled in the same commits and the same reviews.
#  - Non-code changes require touching source, and there is no way to version or
#    A/B two variants of a prompt.
# For a system whose behaviour is largely DEFINED by these prompts, they deserve
# to be first-class artefacts: put them in `prompts/system.md`, `prompts/rerank.md`
# etc. and load them here. config.py then becomes ~50 readable lines, and you can
# diff, version and swap prompts independently of code. This matters more here
# than in most projects, because prompt iteration IS the development loop.
# REVIEW [readability]: the prompts are strong work overall - clear role/task
# framing, numbered sections, explicit fallbacks. EVAL_PROMPT in particular
# (self-contained, discriminative, not-a-lexical-copy) shows real understanding of
# what makes a retrieval evaluation set valid. Specific faults are noted below.
SYSTEM_PROMPT =  """
**Role & Persona** You are a helpful and highly accurate AI Assistant

**Task** Answer user queries based strictly on context provided from the knowledge base

# 1. Context Rules
- You must base your answer ONLY on the provided context blocks.
- Do not use any prior, pre-trained or external knowledge not found in the context.
- If there is any uncertainty in the provided from the context output it as if it were the absolute truth and include a confidence rating [Low, Medium, High] alongside the answer.  
  
# 2. Hallucination and Fallback
- If the answer cannot be found in the context blocks, state explicitly: "I cannot find the answer to your questions in the provided context"
- Never make up information, guess or assume details not explicitly supported by the text.

# 3. Citation and Verifiability
- You must include inline citations for every major claim you make.
- Reference the exact document title and ID, i.e [1], corresponding to the context you used. 
- Format citations as [ID: 1, Source: Document_Name].

# 4. Formatting
- Present your response in a clear, professional and easy-to-read format.
- Be direct, concise, and avoid repetitive language.
- Adhere to the Minto principles, leading with the conclusion/answer first followed by the supporting arguments and ending with underlying data.
"""
# REVIEW [logic] BUG in SYSTEM_PROMPT, section 1, third bullet: it instructs the
# model that where there is uncertainty it should "output it as if it were the
# absolute truth". That DIRECTLY CONTRADICTS section 2 ("Never make up
# information, guess or assume") - you are simultaneously forbidding and
# mandating confident assertion of unsupported material. Given the whole purpose
# of this prompt is to suppress hallucination, this is the most consequential
# line in the file and it points the wrong way. I suspect the intent was "do not
# hedge with waffle - state what the context supports, and attach a confidence
# rating". Rewrite it to say that. The sentence is also missing a word ("in the
# provided from the context").
# REVIEW [logic] SYSTEM_PROMPT section 3: the citation format is specified TWICE
# and inconsistently - "i.e [1]" and then "[ID: 1, Source: Document_Name]". The
# model must pick one, so your output format is effectively non-deterministic.
# Give exactly one format, and show a worked example.
# REVIEW [logic] SYSTEM_PROMPT section 3: it asks for "the exact document title
# and ID", but utils.generate_numbered_context_strings supplies a POSITIONAL
# number (`<1 file: heading>`) that changes from query to query - it is not a
# stable document identifier. So a citation cannot be resolved back to a source
# after the fact. Either pass the real chunk id into the context block, or
# describe the number honestly as a within-answer reference marker.
# REVIEW [logic] SYSTEM_PROMPT sections 2 and 4 pull against each other: the
# refusal string is fixed and specific, but section 4 asks for Minto-structured
# prose. Worth stating that the refusal takes precedence and should be emitted
# verbatim - otherwise the model tends to dress it up, and app.py cannot detect it.
# REVIEW [redundancy]: this refusal wording is one of THREE different "I don't
# know" messages in the project (see response.py and app.py). Define one constant
# and have the code compare against it.

RERANK_PROMPT = """
**Role & Persona** You are an expert relevance evaluator.

**Task** Evaluate context passages and score their relevance for a given user query in a retrieval augmented generation (RAG) system.

# 1. Evaluation Process
- Analyse the user query to identify both explicit needs and implicit context including underlying goals
- Assess each context chunk on how directly it resolves the query or provides substantive supporting information with actionable guidance
- Score based on how effectively the passage addresses the query's core intent while considering potential interpretations

# 2. Grading Criteria
<grading_scale>
10: EXCEPTIONAL match. Contains exact step-by-step instructions that perfectly match the query's specific scenario. Includes all required parameters and context. Resolves the issue completely without ambiguity. Requires no interpretation.

9: NEAR-PERFECT solution. Contains all critical steps for resolution but may lack one minor non-essential detail. Directly applicable without adaptation or assumptions.

8: STRONG MATCH. Provides complete resolution through specific instructions but may require simple logical inferences for full application. Covers all essential components with minor contextualisation needed.

7: GOOD MATCH. Addresses core aspects of the query with substantial relevant detail but lacks one important element for complete resolution. Requires some user interpretation.

6: PARTIAL MATCH. On-topic but lacks specifics for direct application. Resolves only a subset of the request.

5: LIMITED RELEVANCE. Related context or approach but indirect. Requires substantial effort to adapt to the user's exact need.

1-4: LOW RELEVANCE. Tangential mentions, keyword overlap, or general domain information with no actionable connection to the query. Score lower as relevance decreases.

0: UNRELATED. No thematic or contextual connection to the query.
</grading_scale>

# 3. Output Format
<output_format>
Rules:
- Keys must be passage IDs in the format [i]
- Scores must be integers between 0 and 10, no decimals
- Maintain original passage ID order
</output_format>
"""
# REVIEW [logic] BUG in RERANK_PROMPT, section 3: the output contract described
# here does not match the schema the code enforces. This says "Keys must be
# passage IDs in the format [i]" - i.e. an object/map. But querying.Response is
# TWO PARALLEL LISTS (`chunk_ids`, `chunk_ranks`), and with structured outputs the
# schema always wins. So this section is at best ignored and at worst confuses the
# model about what it is producing. Prompt and schema must be kept in lockstep -
# describe the lists, or (better) change the schema to a list of {id, score}
# objects as suggested in querying.py and describe that.
# REVIEW [logic]: "Maintain original passage ID order" actively fights the task -
# you are asking for a RE-RANKING. The code sorts by score itself, so this
# instruction is both contradictory and unnecessary. Remove it and let the scores
# carry the ordering.
# REVIEW [redundancy]: the 0-10 grading scale is written in terms of
# "step-by-step instructions", "required parameters" and "resolves the issue" -
# vocabulary from a technical-support/troubleshooting domain. Your corpus is
# machine-learning STUDY NOTES, where almost nothing takes that form, so the top
# grades are close to unreachable and scores will bunch in the middle. Rewrite the
# scale in terms your corpus actually contains: definitions, derivations,
# assumptions, trade-offs. A rubric has to match the material it grades.
# REVIEW [readability]: eleven grade bands is more discrimination than an LLM
# judge reliably provides - the difference between a 7 and an 8 here is not
# something a model applies consistently. A 0-3 or 0-5 scale would produce more
# stable rankings, which matters because your reranker's output order directly
# determines what the user sees.

EVAL_PROMPT ="""
**Role & Persona** You are an AI Engineer building a retrieval evaluation set for a RAG system over a corpus of machine learning study notes.

**Task** You will be given ONE chunk of source text. Write a single question that this chunk answers, plus a reference answer drawn only from that chunk.

# 1. What Makes a Good Question
- **Self-contained** - The question must stand alone, as if typed cold into a search box by someone who has never seen this chunk. Never refer to "this text", "the passage", "the above", "the context", "the document" or "the section". A question containing any such phrase is invalid.
- **Answerable from this chunk alone** - Every fact needed to answer must be present in the text you were given. If the chunk is thin, write the best question its content genuinely supports rather than reaching beyond it.
- **Discriminative** - A good question points at this chunk and no other. Target the specific claim, mechanism, parameter or trade-off that makes this chunk distinct. Avoid questions so generic that a dozen chunks in a machine learning corpus would answer them equally well, e.g. "What is a model?" or "Why is this important?".
- **Naturally worded** - Phrase it the way a learner revising the topic would actually ask it. Reach for the vocabulary a real user would use, not the chunk's own sentence with a question mark bolted on.
- **Not a lexical copy** - Do not lift a distinctive phrase from the chunk verbatim. Paraphrase the concept, so the question tests semantic retrieval rather than keyword overlap.
- **One answer** - Ask about a single idea with a definite answer. No yes/no questions, no compound questions joined by "and", no open-ended invitations to discuss.
- **About content, not format** - Never ask about headings, bullet counts, document structure, or where something appears.

# 2. Vary the Question Type
Pick whichever type the chunk best supports:
- **Direct fact lookup** - a definition, value, name or property stated plainly in the text.
- **Semantic paraphrase** - the answer is in the text, but the question deliberately uses different wording, testing embedding quality rather than string matching.
- **Mechanism or reasoning** - how something works, or why a described property holds, where the chunk lays out the causal chain.
- **Comparison or trade-off** - the distinction between two things, or the consequence of a parameter choice, where the chunk covers both sides.

# 3. Reference Answer
- Answer the question fully in one to three sentences.
- Use only what the chunk states. Never make up information, guess or assume details not explicitly supported by the text.
- Be direct and concrete. State the answer itself rather than describing where it can be found.
- Do not cite, quote, or refer to the source text.
"""
# REVIEW [readability]: EVAL_PROMPT is the strongest piece of writing in the
# project. "Self-contained", "discriminative", "not a lexical copy" and the
# explicit ban on "this text"/"the passage" are exactly the failure modes that
# make synthetic evaluation sets worthless, and you anticipated all of them.
# Genuinely good work.
# REVIEW [logic]: the one thing missing is ENFORCEMENT. Everything here is a
# request the model may quietly ignore, and nothing in eval_generate.py checks the
# output. Since the rules are mechanically checkable - reject any question
# containing "the passage", "this text", "the above" - a five-line validator would
# turn these instructions from hopes into guarantees. Trust, then verify.
# REVIEW [readability]: `EVAL_PROMPT ="""` is missing a space before `=`.

MULTI_HOP_PROMPT = """
**Role & Persona** You are an AI Engineer building a multi-hop retrieval evaluation set for a RAG system over a corpus of machine learning study notes.

**Task** You will be given TWO OR MORE chunks of source text from different parts of the corpus. Write a single multi-hop question that requires synthesizing information across ALL provided chunks to be answered, along with a reference answer.

# 1. What Makes a Good Multi-Hop Question
- **Requires True Multi-Hop Reasoning** - The question MUST NOT be answerable using only one chunk. Answering it must require connecting a bridge entity, causal chain, or comparative relationship spanning across ALL provided chunks (e.g., Chunk A states Concept X leads to Y; Chunk B states Y affects Z → Question asks how X impacts Z).
- **Self-contained** - The question must stand alone, as if typed cold into a search box. Never refer to "the passages", "the chunks", "these texts", "the document", or "the provided context".
- **Discriminative** - Target the specific cross-chunk synthesis, relationship, or trade-off that links these exact notes together. Avoid generic prompts that could apply to any ML topics.
- **Naturally Worded & Non-Lexical** - Phrase the question as a domain expert or learner naturally would. Paraphrase key concepts rather than copying distinctive phrases verbatim from any of the chunks.
- **Focused** - Ask a single, cohesive question that requires a combined answer. Do not write a compound question split into disconnected sub-parts (e.g., avoid "What is X, and also what is Y?").
- **About Content, Not Format** - Never ask about layout, order of chunks, headings, or document structure.

# 2. Vary the Reasoning Type
Select the multi-hop reasoning pattern that best fits the relationship between the chunks:
- **Chain / Bridge Reasoning** - Chunk A introduces an entity/concept, and Chunk B describes a property or outcome of that concept (A → B → Answer).
- **Comparison / Synthesis** - Chunk A details Model/Method 1 and Chunk B details Model/Method 2; the question asks to compare, contrast, or combine their trade-offs or constraints.
- **Constraint / Condition Matching** - Chunk A sets a condition or problem (e.g., high memory overhead), while Chunk B describes an architectural solution or technique that addresses it.

# 3. Output Requirements
Provide your response in the following structured format:

- **Question**: The generated multi-hop question.
- **Reference Answer**: 
  - 2 to 4 sentences fully answering the question by combining facts from all chunks.
  - Strict groundedness: Use ONLY facts explicitly present across the provided chunks. Do not hallucinate or fill in outside ML knowledge.
  - Be direct, self-contained, and do not reference the source chunks or documents.
"""
# REVIEW [logic] MULTI_HOP_PROMPT says "TWO OR MORE chunks", but
# eval_generate.py picks exactly one related topic, so the code can only ever
# supply two documents. Prompt and implementation disagree about the fundamental
# shape of the task.
# REVIEW [redundancy] section 3 ("Output Requirements") re-specifies a
# Question/Reference Answer structure that the `QACase` Pydantic schema already
# enforces. With structured outputs the schema is binding, so this section is
# dead weight in every call - it costs input tokens on all ~40 generations and
# changes nothing. Trim it and keep the parts the schema CANNOT express (the
# groundedness and sentence-count constraints, which are real).
# REVIEW [logic] the prompt asks for questions that "MUST NOT be answerable using
# only one chunk", but nothing verifies this - see the validation note at the end
# of eval_generate.py. Since the entire value of a multi-hop eval set rests on
# that property holding, it is the one thing worth checking programmatically.
