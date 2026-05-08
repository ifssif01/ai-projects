# RAG Chatbot for Car Manual Documentation

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about MG ZS car warning messages using an HTML car manual as its knowledge base.

## What it does

Loads an HTML car manual, splits it into chunks, stores them as embeddings in a vector store, and uses GPT-4o-mini to answer questions based on the retrieved context.

## Stack

- [LangChain](https://www.langchain.com/) — document loading, splitting, and RAG chain
- `gpt-4o-mini` — answer generation
- `text-embedding-3-small` — document embeddings
- [Chroma](https://www.trychroma.com/) — vector store

## Requirements

- Python 3.x
- Required packages:

```bash
pip install langchain langchain-openai langchain-community langchain-chroma langchain-text-splitters unstructured
```

- An OpenAI API key set as an environment variable:

```bash
export OPENAI_API_KEY=your_key_here
```

- The car manual HTML file placed at:

```
data/mg-zs-warning-messages.html
```

## Usage

```bash
python Building_RAG_Chatbots_for_Technical_Documentation.py
```
