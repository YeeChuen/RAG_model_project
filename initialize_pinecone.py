# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:01:34 2024

@author: kpall
"""

import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
from config import Config

def initialize_pinecone(vector_dimension=384):
    
    pc = Pinecone(api_key=Config.API_KEY)
    print(Config.API_KEY)
   
    '''if Config.INDEX_NAME not in pinecone.list_indexes().names():
        pinecone.create_index(name=Config.INDEX_NAME, dimension=384'''  # Adjust dimension as needed
    return pc.Index(Config.INDEX_NAME)
   
    


