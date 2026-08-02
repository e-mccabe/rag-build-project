"""Provider-agnostic LLM calls: one interface over both Anthropic and OpenAI.

Callers pass a 'provider:model' string and get back plain text.
"""

from collections.abc import Iterator

from pydantic import BaseModel

from rag_build.config import MODELS, Models, get_anthropic_client, get_openai_client


class AiClient:
    """Calls the configured model for each pipeline role.

    Holds the model IDs and token budget so callers pass prompts, not model strings.
    """

    def __init__(self,models:Models,max_tokens:int):

        self.embedding = models.embedding
        self.response = models.response
        self.reranking = models.reranking
        self.max_tokens = max_tokens

    def generate_text(self,message:str,system:str) -> str:
        """Generate a complete LLM response as a single string.

        Blocks until the model has finished, then returns the whole answer.
        """
        provider, model = self._split_model(self.response)

        if provider == 'anthropic':
            return self._get_anthropic_text(model,self.max_tokens,message,system)

        if provider == 'openai':
            return self._get_openai_text(model,self.max_tokens,message,system)

        raise ValueError(f'No model found for {self.response}, enter a valid anthropic or openai model, in the format openai:gpt4o')

    def generate_stream(self,message:str,system:str) -> Iterator[str]:
        """Generate an LLM response as a stream of text chunks.

        Concatenating every chunk gives the same string generate_text() returns.
        """
        provider, model = self._split_model(self.response)

        # elif/else rather than three separate ifs: `yield from` resumes on the next
        # line once the stream is exhausted, so an unguarded `raise` below would fire
        # at the end of every successful call. Being a generator, that raise reaches
        # the caller at their loop rather than at the call.
        if provider == 'anthropic':
            yield from self._get_anthropic_stream(model,self.max_tokens,message,system)

        elif provider == 'openai':
            yield from self._get_openai_stream(model,self.max_tokens,message,system)

        else:
            raise ValueError(f'No model found for {self.response}, enter a valid anthropic or openai model, in the format openai:gpt4o')

    def generate_rerank(self,message:str,system:str,structure:type[BaseModel]) -> BaseModel | None:
        """Generate a reranking response parsed into the given Pydantic schema.

        Returns None if the model's output could not be parsed into that schema.
        """
        provider, model = self._split_model(self.reranking)

        if provider == 'anthropic':
            return self._get_anthropic_structured(model,self.max_tokens,message,system,structure)

        if provider == 'openai':
            return self._get_openai_structured(model,self.max_tokens,message,system,structure)

        raise ValueError(f'No model found for {self.reranking}, enter a valid anthropic or openai model, in the format openai:gpt4o')

    def generate_eval_case(self,message:str,system:str,structure:type[BaseModel]) -> BaseModel | None:
        """Generate a evaluation response parsed into the given Pydantic schema.

        Returns None if the model's output could not be parsed into that schema.
        """
        provider, model = self._split_model(self.response)

        if provider == 'anthropic':
            return self._get_anthropic_structured(model,self.max_tokens,message,system,structure)

        if provider == 'openai':
            return self._get_openai_structured(model,self.max_tokens,message,system,structure)

        raise ValueError(f'No model found for {self.response}, enter a valid anthropic or openai model, in the format openai:gpt4o')

    def generate_embeddings(self,input_text:list[str]) -> list[list[float]]:
        """Embed many strings in one API call, returning one vector per input.

        Only OpenAI is supported, as Anthropic has no embeddings endpoint.
        """
        provider, model = self._split_model(self.embedding)

        if provider != 'openai':
            raise ValueError(f'Embedding Model must be openai: {self.embedding}')

        response = get_openai_client().embeddings.create(model=model,input=input_text)

        return [item.embedding for item in response.data]

    @staticmethod
    def _split_model(generation_model:str) -> tuple[str,str]:
        """Split a 'provider:model' string into its two parts.

        The provider is lowercased for matching; the model is left as typed because
        some API model IDs are case-sensitive.
        """
        try:
            provider, model = generation_model.split(':')
        except ValueError:
            raise ValueError(f"Incorrect model string: {generation_model}, input as 'provider:model' (e.g: 'openai:gpt-4o')")

        return provider.strip().lower(), model.strip()

    @staticmethod
    def _get_anthropic_text(model:str,max_tokens:int,message:str,system:str) -> str:
        """Call an Anthropic model and return the text of its response."""

        response = get_anthropic_client().messages.create(
            model= model,
            max_tokens= max_tokens,
            system= system,
            messages= [
                {"role":"user","content":message}
            ]
        )

        # response.content is a list of blocks, not a string: one reply can mix text
        # with thinking or tool-use blocks, and only text blocks carry a .text
        # attribute. Filter to those and join, since prose can span several blocks.
        return ''.join(block.text for block in response.content if block.type == 'text')

    @staticmethod
    def _get_openai_text(model:str,max_tokens:int,message:str,system:str) -> str:
        """Call an OpenAI model and return the text of its response."""

        response = get_openai_client().chat.completions.create(
            model = model,
            max_completion_tokens=max_tokens,
            messages= [
                {"role":"system","content":system},
                {"role":"user","content":message}
            ]
        )

        # choices[0] because we only asked for one completion. or '' guards the
        # None that comes back when a response is cut short by the content filter.
        return response.choices[0].message.content or ''

    @staticmethod
    def _get_anthropic_stream(model:str,max_tokens:int,message:str,system:str) -> Iterator[str]:
        """Stream an Anthropic model's response as text chunks."""

        with get_anthropic_client().messages.stream(
                model = model,
                max_tokens=max_tokens,
                system= system,
                messages= [
                    {'role':'user','content':message}
                ]
            ) as stream:
                yield from stream.text_stream

    @staticmethod
    def _get_openai_stream(model:str,max_tokens:int,message:str,system:str) -> Iterator[str]:
        """Stream an OpenAI model's response as text chunks."""

        stream = get_openai_client().chat.completions.create(
            model = model,
            max_completion_tokens= max_tokens,
            messages= [
                {'role':'system','content':system},
                {'role':'user','content':message}
            ],
            stream= True
        )

        # Each chunk carries a delta - the new text since the previous one. The
        # first and last chunks announce the role and finish reason with no text, so
        # their content is None; skip those rather than yielding empty strings.
        for piece in stream:
            content = piece.choices[0].delta.content
            if content:
                yield content

    @staticmethod
    def _get_openai_structured(model:str,max_tokens:int,message:str,system:str,structure:type[BaseModel]) -> BaseModel | None:
        """Call an OpenAI model and return its response parsed into the given schema."""

        response = get_openai_client().chat.completions.parse(
            model = model,
            max_completion_tokens= max_tokens,
            messages= [
                {'role':'system','content':system},
                {'role':'user','content':message}
            ],
            response_format= structure
        )

        # parsed is None when the model refuses or the output fails validation.
        return response.choices[0].message.parsed

    @staticmethod
    def _get_anthropic_structured(model:str,max_tokens:int,message:str,system:str,structure:type[BaseModel]) -> BaseModel | None:
        """Call an Anthropic model and return its response parsed into the given schema."""

        response = get_anthropic_client().messages.parse(
            model = model,
            max_tokens= max_tokens,
            system= system,
            messages= [
                {'role':'user','content':message}
            ],
            output_format= structure
        )

        return response.parsed_output

AI = AiClient(MODELS, max_tokens=500)