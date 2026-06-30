# RAG Pipeline for Markdown Second Brain

A question-answering chat bot over a collection of Machine Learning notes. 
Built as simple RAG pipeline in Python and Streamlit that uses markdown files to improve OpenAI API querying context.

Ask a question in plain english and get an answer grounded in the notes.

#### Web-app Demo




## Table of Contents
- Quick Start
- Features
- Installation
- Usage
- Tech Stack
- Contributing 
- License


## Quickstart

Run the project locally in four steps. 
Requires:
**Python 3.13** and a paid **OpenAI** API key

```bash
git clone https://github.com/e-mccabe/rag-build-project.git
cd rag-build-project
uv sync # install dependencies
cp .env.example .env # paste API key into .env
uv run streamlit run app.py # opens app in browser
```
To use a different corpus input your own `.md` files in `/data`.

## Features

A three-phase pipeline. **Indexing** turns notes into searchable vectors; **Retrieval** retrieves and ranks information from the searchable database of vectors; **Generation** uses the retrieved information, alongside LLM to generate a response to the input query  

- **Loading** (*src/loading.py*) - load each note to provide the path, file name, metadata and content for each.  
- **Chunking** (*src/chunking.py*) - split each document using headings to create chunks, keeping metadata.
- **Embedding** (*src/embeding.py*) - turn each chunk into a vector and store it in a local Chroma.
- **Retrieval** (*src/querying.py*) - find the closest chunks to the input query and re-order them with an LLM for relevance
- **Generate** (*src/response.py*) - Provide the LLM with the most relevant/top chunks, a system prompt and the input query. It answers using only those, using citations to the original documents

### Configuration & Limitations

- Tunable in `src/config.py`: LLM models, system prompts, evaluation sets
- **This is a portfolio project demo by design** made for single-user, no auth, small corpus, local on-disk Chroma.

### Architecture Flow 


```mermaid
graph TD;
    A[Document Corpus] --> B[Loading and Chunking];
    B --> C[Embedding Chunked Corpus];
    C --> D(Vector Store);
    D --> E[Retrieve Top-K Chunks];
    F((User Query)) --> G[Embed Query];
    G --> E;
    E --> H(Rerank via LLM);

    subgraph ParentBox[LLM Context]
        H(Rerank via LLM)
        L(LLM Generates Response)
    end

    F((User Query)) --> I[Build Prompt];
    H -->|Inject Selected Chunks| I[Build Prompt];
    I -->|Context & User Query| L(LLM Generates Response);
    K(System Prompt) --> I;
    L --> J(Response Output);

    style A fill:#FCEBEB,stroke:#A32D2D,color:#501313;
    style D fill:#EAF3DE,stroke:#3B6D11,color:#173404;
    style F fill:#185FA5,stroke:#0C447C,color:#E6F1FB;
    style H fill:#FAEEDA,stroke:#854F0B,color:#412402;
    style L fill:#FAEEDA,stroke:#854F0B,color:#412402;
    style J fill:#EAF3DE,stroke:#3B6D11,color:#173404;
    style K fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A;
    style ParentBox fill:#FFF3EE,stroke:#854F0B,color:#412402;
```

### Evaluation

| Stage | Purpose | Evaluation Dataset | Evaluation Focus |
|-------|---------|--------------------|------------------|
|Development|Experimenting with inputs and system design|Curated offline question sets|Retrieval Accuracy, Response Correctness|
|Stress Testing|Robustness and safety for production|Edge Cases, red-teamed prompts|Prompt injection resistance, hallucination checks|
|Monitoring|Measure UX, and detect poor performance|Live Queries|Completeness, safety, refusals|
|Regression Testing|Prevent silent failures after updates|Curated offline question sets|Test risky scenarios|


## Installation

## Usage 


## Tech Stack


## Contributing


## License

MIT.

## Elements to be added

- Large Paragraphs logic
- wikilink used as metadata
- Error handling when extracting expect JSON block from LLM in rerank()

