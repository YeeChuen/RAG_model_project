# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:01:42 2024

@author: kpall
"""

def index_embeddings(index, chunks, embeddings):
    assert len(chunks) == len(embeddings), "Chunks and embeddings length mismatch"
    # Creating a list of tuples (id, vector)
    vectors = [(str(i), embedding.tolist()) for i, embedding in enumerate(embeddings)]
    # Upserting vectors into the Pinecone index
    index.upsert(vectors=vectors)
