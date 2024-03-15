# Author(s): Yee
'''
Usage:
    python3 upload.py --pdf_file=Correcting_Diverse_Factual_Errors_paper.pdf --pinecone_key=
'''
#_____
# imports
import argparse
from tqdm import tqdm

# document loader
from langchain_community.document_loaders import PyPDFLoader
# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# text embedding
from sentence_transformers import SentenceTransformer

# store to a vector DB
from pinecone import Pinecone

#_____
# variables
result_file = "result.txt"


#_____
# functions
def getArgument():
    parser = argparse.ArgumentParser(description="Upload a PDF file to the RAG model.")
    parser.add_argument("--pdf_file", help=":Path to the PDF file to upload", required=True)
    parser.add_argument("--pinecone_key", help=":The API key in your Pinecone console at https://app.pinecone.io/", default = '')
    #parser.add_argument("--openai_key", help=":The API key for open-ai", default = '')
    return parser.parse_args()

def loadPdf(pdf_file):
    print("Loading PDF file...")
    loader = PyPDFLoader(pdf_file) # <-- basic pypdf, not going to extract images. (Does not work with my old local Mac)
    document = loader.load_and_split()
    return document

def textSplitter(document):
    print('Spltting document...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    split_texts = text_splitter.split_documents(document)
    return split_texts

def textEmbedding(split_texts):
    print('Embedding document...')
    model = SentenceTransformer("all-MiniLM-L6-v2") # <-- smaller embedding model

    vectors = []
    for text in tqdm(split_texts):
        vector = model.encode([text.page_content]) 
        vectors.append(vector[0])

    return vectors

def upsert(index_db, api_key, vectors, split_texts):
    print("Adding to a temporary vector db...")
    pc = Pinecone(api_key = api_key)
    index = pc.Index(index_db) # <-- create a index in pinecone.io first
    
    upserts_list = []
    for i in tqdm(range(len(vectors))):
        vector = vectors[i]
        split_text = split_texts[i]
        data = {}
        metadata = {}
        metadata['page_content'] = split_text.page_content
        metadata['source'] = split_text.metadata["source"]
        metadata['page'] = split_text.metadata["page"]

        data["id"] = f"{metadata['source']}_p{metadata['page']}_{i}"
        data["values"] = vector
        data["metadata"] = metadata

        upserts_list.append(data)

    index.upsert(vectors = upserts_list)

def main():
    args = getArgument()

    document = loadPdf(args.pdf_file)
    split_texts = textSplitter(document)    
    vectors = textEmbedding(split_texts)

    # upload to pinecone db
    upsert("upload-db", args.pinecone_key, vectors, split_texts)

#_____
# main
if __name__ == "__main__":
    main()
