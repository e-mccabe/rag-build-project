from openai import OpenAI

from src.config import EMBEDDING_MODEL, RESPONSE_MODEL


def check_openai() -> None:

    client = OpenAI()
    embed = client.embeddings.create(model=EMBEDDING_MODEL,input='hello world')
    dims = len(embed.data[0].embedding)
    print(f'OpenAI working - test string embedded in {dims}-dimensional vector')

    resp = client.responses.create(model =RESPONSE_MODEL,
                                   max_output_tokens=20,
                                   input = 'Repy with confirmation model set up is successful')
    
    print(f'Open AI {RESPONSE_MODEL} works. Response from model {resp}')


if __name__ == "__main__":
    check_openai()
        
