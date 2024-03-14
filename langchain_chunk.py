# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:53:34 2024

@author: kpall
"""

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import TextSplitConfig


def chunk_text(text):
    if TextSplitConfig.TYPE == "character":
        splitter = CharacterTextSplitter(chunk_size=TextSplitConfig.CHUNK_SIZE, chunk_overlap=TextSplitConfig.OVERLAP, separator='')
    elif TextSplitConfig.TYPE == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=TextSplitConfig.CHUNK_SIZE, chunk_overlap=TextSplitConfig.OVERLAP)
    elif TextSplitConfig.TYPE == "token":
        splitter = RecursiveCharacterTextSplitter.from_token_encoder(chunk_size=TextSplitConfig.CHUNK_SIZE, chunk_overlap=TextSplitConfig.OVERLAP)
    else:
        raise ValueError("Invalid text split type specified in configuration.")
    
    chunks = splitter.split_text(text)
    return chunks