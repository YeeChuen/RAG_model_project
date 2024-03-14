# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:01:27 2024

@author: kpall
"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(chunks):
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return embeddings
