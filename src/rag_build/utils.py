
def generate_numbered_context_strings(hits:list[dict]) -> str:
    """
    Takes the resulting list of chunks & their meta and creates a single string to pass LLM
    
    The output string is in the form

    '<0 example_file: Heading 1 > Heading 2 >

    Example Text

    <1 next_file: Heading 1 > Heading 2 >

    Example Text'
    """
    context_strings = []

    for i, hit in enumerate(hits,1):

        file = hit['metadata']['file']
        headings = hit['metadata']['headings'].split(',')
        breadcrumb = f'<{i} {file}: {' > '.join(headings)}>'
        full_text = f'{breadcrumb}\n\n{hit['text']}' 
        context_strings.append(full_text)

    return '\n\n'.join(context_strings)
