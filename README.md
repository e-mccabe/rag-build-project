# RAG Pipeline for Markdown Second Brain

A simple RAG pipeline built in Python and Streamlit that uses markdown files from Obsidian second brain for to improve OpenAI API querying context.

## Table of Contents
- Features
- Installation
- Usage
- Tech Stack
- Contributing 
- License

## Features

- **Loading**
- **Chunking** 
- **Embedding**
- **Vector Storing & Search**
- **Querying**


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

## Elements to be added

- Large Paragraphs logic
- wikilink used as metadata
- Error handling when extracting expect JSON block from LLM in rerank()

