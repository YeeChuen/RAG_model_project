# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:01:49 2024

@author: kpall
"""

from extract import extract_text_from_pdf
from langchain_chunk import recursive

from embed import embed_text
from initialize_pinecone import initialize_pinecone
from index_embeddings import index_embeddings

import sys

def process_pdf(pdf_path):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
 
    # Chunk text
    chunks = recursive(text)
    
    # Embed chunks
    embeddings = embed_text(chunks)
  
    
    # Initialize Pinecone
    index = initialize_pinecone()
    
    # Index embeddings
    index_embeddings(index, chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks from '{pdf_path}'")

if __name__ == "__main__":
   if len(sys.argv) != 2:
        print("Usage: python script_name.py <pdf_file_path>")
        sys.exit(1)

   pdf_path = sys.argv[1]  # Get PDF file path from command-line argument
   process_pdf(pdf_path)