# RAG System with LangChain and Pinecone

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain for text processing and Pinecone for vector indexing. The system allows users to upload PDF files, clean the text, split it into chunks, embed the chunks, and store them in Pinecone for efficient retrieval.

## Introduction

The RAG system utilizes LangChain for chunking, Sentence Transformers for embedding, while Pinecone is used as a vector database for efficient retrieval. This README provides instructions on how to install and use the system.

## Installation

1. Clone the repository:

2. Navigate to the project directory

3. Install the required dependencies


## Usage

1. Ensure you have LangChain and Pinecone API key set up. 
2. Run the script with the path to the PDF file as a command-line argument:

    ```bash
    python main_script.py path/to/your/pdf.pdf
    ```

## Configuration

You can customize the text splitting parameters and other settings by modifying the `config.py` file.

