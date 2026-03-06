# Week 15 Graded Mini Project – RAG Assistant

## Domain
Technology – Git Documentation Assistant

## Objective
Build a Retrieval-Augmented Generation (RAG) assistant that answers questions using only provided documents.

## Features
- Document ingestion and chunking
- OpenAI embeddings
- FAISS vector database
- Retrieval-Augmented generation
- Refusal when answer is not in documents
- Conversation-style interaction

## Data Sources
Example sources used:

- https://git-scm.com/docs
- https://docs.github.com/en/get-started

Documents were converted to TXT format and stored locally.

## Setup

Install dependencies:

pip install langchain langchain-community langchain-openai faiss-cpu

Set API key:

export OPENAI_API_KEY="your_api_key"

Run ingestion:

python ingest.py

Start chatbot:

python chatbot.py

## Example Questions

- What is a pull request?
- What is a Git branch?
- What are merge conflicts?
- Does Git deploy code automatically?

## Safety Behavior

If the answer is not in the documents the assistant responds with:

"I don’t have enough information in the provided documents."

## Files

- ingest.py
- chatbot.py
- README.md
- sample_conversation.txt
- Jupyter Notebook
