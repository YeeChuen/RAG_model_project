## RAG model

Retrieval-augmented generation (RAG) is a framework to give generative models knowledge without finetuning themselves. In this way, an LLM can adapt to new tasks quickly with the presence of new documents.

### Summary

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain for text processing and Pinecone for vector indexing. The system allows users to upload PDF files, clean the text, split it into chunks, embed the chunks, and store them in Pinecone for efficient retrieval.

The RAG system utilizes LangChain for chunking, Sentence Transformers for embedding, while Pinecone is used as a vector database for efficient retrieval. This README provides instructions on how to install and use the system.

### Goal

To acquire the knowledge and understanding of building a RAG model with utilizing existing pipelines and library to assist in this implementation. Moreover, experimenting with various paramters that affects the performance of RAG model such as different chunking size, size of top-k chunk retrived to be feed into LLM and so on.

### Future

Future Improvement to be updated.

### Installation
Follow the steps below for installation:

1. Clone the repository

```
git clone https://github.com/YeeChuen/RAG_model_project
```

2. Navigate to the project,

```
cd ./RAG_model_project
```

3.0 **Ensure that you have enable long path on your Windows, follow steps [Microsoft Tutorial](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry) or [Video guide](https://www.youtube.com/watch?v=E0e28Z1iHWs)

3.1 Install python dependencies

```
pip install -r requirements.txt
```

4. Download up Ollama following on https://github.com/ollama/ollama, then download llama2 model using 

```
ollama run llama2
```

6. Load model ```llama2```

```
ollama pull llama2
```

5. Set up Pinecone account on https://app.pinecone.io/

6. Create an index in your Pinecone account, note: use dimension = 384.

### Usage

1. Ensure that requirements is installed, and dependencies has been set up(Ollama, Pinecone, etc).

2. Access your Pinecone API key in the "API Keys" section in the side bar on https://app.pinecone.io/ after login.

3. upload your pdf file using 

```
python3 upload.py --pdf_file=<path/to/PDF> --pinecone_key=<Pinecone API key>
```

### Configuration

No configurations yet.
