from openai import OpenAI

from src.config import EMBEDDING_MODEL, RESPONSE_MODEL


def check_openai() -> None:

    client = OpenAI()
    embed = client.embeddings.create(model=EMBEDDING_MODEL,input='hello world')
    dims = len(embed.data[0].embedding)
    print(f'OpenAI working - test string embedded in {dims}-dimensional vector')

    resp = client.responses.create(model =RESPONSE_MODEL,
                                   max_output_tokens=20,
                                   input = 'Reply with confirmation model set up is successful')
    
    print(f'OpenAI {RESPONSE_MODEL} works. Model response: {resp.output_text}')


if __name__ == "__main__":
    check_openai()
        
