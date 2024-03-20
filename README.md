# RAG model

Retrieval-augmented generation (RAG) is a framework to give generative models knowledge without finetuning themselves. In this way, an LLM can adapt to new tasks quickly with the presence of new documents.

## Summary

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain for text processing and Pinecone for vector indexing. The system allows users to upload PDF files, clean the text, split it into chunks, embed the chunks, and store them in Pinecone for efficient retrieval.

The RAG system utilizes LangChain for chunking, and Sentence Transformers for embedding, while Pinecone is used as a vector database for efficient retrieval. This README provides instructions on how to install and use the system.

## Goal

To acquire the knowledge and understanding of building a RAG model by utilizing existing pipelines and libraries to assist in this implementation. Moreover, experimenting with various parameters that affect the performance of the RAG model such as different chunking sizes, size of top-k chunk retrieved to be fed into LLM, and so on.

## Future

Future Improvements to be updated.

## Installation
Follow the steps below for installation:

1. Clone the repository

```
git clone https://github.com/YeeChuen/RAG_model_project
```

2. Navigate to the project,

```
cd ./RAG_model_project
```

3. **Ensure that you have enabled long path on your Windows, follow steps [Microsoft Tutorial](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry) or [Video guide](https://www.youtube.com/watch?v=E0e28Z1iHWs)

4. Install python dependencies

```
pip install -r requirements.txt
```

5. Download up Ollama following on [Ollama](https://github.com/ollama/ollama), then download llama2 model using 

```
ollama run llama2
```

6. Load model ```llama2```

```
ollama pull llama2
```

7. Set up a Pinecone account on [Pinecone](https://app.pinecone.io/).

8. Create an index in your Pinecone account, ** note: use index name = 'rag-db', dimension = 384.

## Usage

1. Ensure that requirements are installed, and dependencies have been set up(Ollama, Pinecone, etc).

2. Access your Pinecone API key in the "API Keys" section in the sidebar on [Pinecone](https://app.pinecone.io/) after login.

3. upload your PDF file using 

```
python upload.py --pdf_file=<path/to/PDF> --pinecone_key=<Pinecone API key>
```

## Configuration

Configurations to be updated.
