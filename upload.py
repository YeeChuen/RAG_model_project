# Author(s): Yee
'''
Usage:
    python3 upload.py --pdf_file=Correcting_Diverse_Factual_Errors_paper.pdf --pinecone_key=
'''
#_____
# imports
import argparse
# document loader
from langchain_community.document_loaders import PyPDFLoader
# text splitter
from langchain_text_splitters import CharacterTextSplitter
# text embedding
from langchain_community.embeddings import OllamaEmbeddings
# store to a vector DB
from pinecone import Pinecone

# testing
from langchain_community.llms import Ollama

#_____
# variables

#_____
# functions
def getArgument():
    parser = argparse.ArgumentParser(description="Upload a PDF file to the RAG model.")
    parser.add_argument("--pdf_file", help=":Path to the PDF file to upload", required=True)
    parser.add_argument("--pinecone_key", help=":The API key in your Pinecone console at https://app.pinecone.io/", required=True)
    return parser.parse_args()

def loadPdf(pdf_file):
    loader = PyPDFLoader(pdf_file) # <-- basic pypdf, not going to extract images. (Does not work with my old local Mac)
    document = loader.load_and_split()
    #print(len(document)) # <-- development and debug purposes
    #print(document[0]) # <-- development and debug purposes
    return document

def textSplitter(document):
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    split_texts = text_splitter.split_documents(document)
    #print(split_texts[0]) # <-- development and debug purposes
    #print(len(split_texts)) # <-- development and debug purposes
    return split_texts

def textEmbedding(split_texts):
    embeddings = OllamaEmbeddings(model="llama2")

    # testing 
    print("Testing on random text")
    random_text = "test ollama embedding result"
    print(random_text) # <-- development and debug purposes
    query_result = embeddings.embed_query(random_text)
    #print(query_result) # <-- development and debug purposes
    print(len(query_result)) # <-- development and debug purposes

    # testing 
    print("Testing on 1 of the split_text")
    print(split_texts[0]) # <-- development and debug purposes
    query_result = embeddings.embed_query(split_texts[0])
    #print(query_result) # <-- development and debug purposes
    print(len(query_result)) # <-- development and debug purposes

    print("Embedding the whole of split_texts")
    vector = embeddings.embed_documents(split_texts)
    print(vector) # <-- development and debug purposes
    print(len(vector)) # <-- development and debug purposes


def main():
    args = getArgument()

    print("Spawnning a temporary vector db... (TODO)")
    pc = Pinecone(api_key=args.pinecone_key)
    #index = pc.Index("tempdb") # <-- TODO: implement pinecone free version, Starter Index.

    print("Loading PDF file...")
    document = loadPdf(args.pdf_file)

    print('Spltting document...')
    split_texts = textSplitter([document[0]]) # <-- testing only, hence we are only splitting and embedding page 1.
    
    print('Embedding document...')
    textEmbedding(split_texts)

    #pc.delete_index("tempdb") # <-- TODO: delete the starter index created for testing Upload.py

#_____
# main
if __name__ == "__main__":
    main()
