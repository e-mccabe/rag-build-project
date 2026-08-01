
from rag_build.config import get_anthropic_client, get_openai_client


def generate_text(generation_model:str,max_tokens:int,message:str,system:str) -> str:
    """
    Generates LLM response as a string regardless of provider and model
    
    generation_model must be a string of format 'provider:model'
    example: anthropic:claude-opus-5
    """
    provider, model = generation_model.split(':')

    if provider.lower() == 'anthropic':
        result = _get_anthropic_text(model,max_tokens,message,system)

    elif provider.lower() == 'openai':
        result = _get_openai_text(model,max_tokens,message,system)

    else:
        raise ValueError(f'No model found for {generation_model}, enter a valid anthropic or openai model, in the format openai:gpt4o')
    
    return result




def _get_anthropic_text(model:str,max_tokens:int,message:str,system:str) -> str:
    """Generates response from antropic model and return the content text"""

    message = get_anthropic_client().messages.create(
        model= model,
        max_tokens= max_tokens,
        system= system,
        messages= [
            {"role":"user","content":message}
        ] 
    )

    return message.content[0].text

def _get_openai_text(model:str,max_tokens:int,message:str,system:str) -> str:

    message = get_openai_client().chat.completions.create(
        model = model,
        max_completion_tokens=max_tokens,
        messages= [
            {"role":"system","content":system},
            {"role":"user","content":message}
        ]
    )

    return message.choices[0].message.content

if __name__ == '__main__':
    model_1 = 'anthropic:claude-haiku-4-5-20251001'
    model_2 = 'openai:gpt-4o'
    system = 'Answer the question as Hunter S. Thompson. Keep answers to one sentence'
    message = 'What is your favourite holiday memory?'
    print(f'Model = {model_1}\n{generate_text(model_1,250,message,system)}\n\n')
    print(f'Model = {model_2}\n{generate_text(model_2,250,message,system)}\n\n')