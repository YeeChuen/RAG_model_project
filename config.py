# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:30:15 2024

@author: kpall
"""

# config.py

class Config:
    API_KEY = "1aca6064-fadd-4606-837a-5651524ceeb9"
    INDEX_NAME = "ragindex"


# config.py

class TextSplitConfig:
    TYPE = "recursive"  # Specify the type of text splitting: "character", "recursive", "token"
    CHUNK_SIZE = 150  # Specify the chunk size
    OVERLAP = 10  # Specify the overlap
