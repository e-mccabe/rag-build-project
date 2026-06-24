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

-**Loading**
-**Chunking** 
-**Embedding**
-**Vector Storing & Search**
-**Querying**

### Architecture Flow 

```mermaid
graph TD;
    Document Corpus -> Loading & Chunking;
    Loading & Chunking -> Embedding Chunked Corpus;
    Vector Store -> Retrieve Top-K Chunks;
    User Query -> Embed Query;
    Embed Query -> Retrieve Top-K Chunks;
    Retrieve Top-K Chunks -> LLM;
    LLM -> Build Prompt;
    Build Prompt -> LLM;
    LLM -> Response Output;
```

## Installation

## Usage 


## Tech Stack


## Contributing


## License

