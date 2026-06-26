

def _generate_numbered_context_strings(hits:list[dict]) -> str:

    context_strings = []

    for i, hit in enumerate(hits,1):

        file = hit['metadata']['file']
        headings = hit['metadata']['headings'].split(',')
        breadcrumb = f'<{i} {file}: {' > '.join(headings)}>'
        full_text = f'{breadcrumb}\n\n{hit['text']}' 
        context_strings.append(full_text)

    return '\n\n'.join(context_strings)
