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
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings # <-- testing
# store to a vector DB
from pinecone import Pinecone

# testing
from langchain_community.llms import Ollama

#_____
# variables
result_file = "result.txt"


#_____
# functions
def getArgument():
    parser = argparse.ArgumentParser(description="Upload a PDF file to the RAG model.")
    parser.add_argument("--pdf_file", help=":Path to the PDF file to upload", required=True)
    parser.add_argument("--pinecone_key", help=":The API key in your Pinecone console at https://app.pinecone.io/", default = '')
    parser.add_argument("--openai_key", help=":The API key for open-ai", default = '')
    return parser.parse_args()

def loadPdf(pdf_file):
    print("Loading PDF file...")
    loader = PyPDFLoader(pdf_file) # <-- basic pypdf, not going to extract images. (Does not work with my old local Mac)
    document = loader.load_and_split()
    #print(len(document)) # <-- development and debug purposes
    #print(document[0]) # <-- development and debug purposes
    return document

def textSplitter(document):
    print('Spltting document...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    split_texts = text_splitter.split_documents(document)
    print(split_texts[0]) # <-- development and debug purposes
    print(split_texts[0].page_content) # <-- development and debug purposes
    print(split_texts[0].metadata["source"]) # <-- development and debug purposes
    print(split_texts[0].metadata["page"]) # <-- development and debug purposes
    #print(split_texts[1]) # <-- development and debug purposes
    #print(split_texts[2]) # <-- development and debug purposes
    #print(split_texts[3]) # <-- development and debug purposes
    #print(split_texts[4]) # <-- development and debug purposes
    #print(len(split_texts)) # <-- development and debug purposes
    return split_texts

def textEmbedding(split_texts):
    print('Embedding document...')
    embeddings = OllamaEmbeddings(model="llama2")

    '''
    # testing 
    print("Testing on random text")
    random_text = "test ollama embedding result"
    print(random_text) # <-- development and debug purposes
    query_result = embeddings.embed_query(random_text) # <-- vector size of 4096
    #print(query_result) # <-- development and debug purposes
    print(len(query_result)) # <-- development and debug purposes
    '''
    '''
    # testing 
    print("Testing on 1 of the split_text")
    print(split_texts[0]) # <-- development and debug purposes
    query_result = embeddings.embed_query(split_texts[0]) # <-- vector size of 4096
    #print(query_result) # <-- development and debug purposes
    print(len(query_result)) # <-- development and debug purposes
    '''
    print("Embedding the whole of split_texts")
    vectors = []
    debug = 0
    for text in tqdm(split_texts):
        if debug == 5: break
        vectors.append(embeddings.embed_query(text))
        debug += 1

    #print(vectors) # <-- development and debug purposes
    #print(len(vectors)) # <-- development and debug purposes
    #print(len(vectors[0])) # <-- development and debug purposes
    #print(len(vectors[1])) # <-- development and debug purposes
    #print(len(vectors[2])) # <-- development and debug purposes
    #print(len(vectors[3])) # <-- development and debug purposes
    #print(len(vectors[4])) # <-- development and debug purposes

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

    '''
    index.upsert(
    vectors=[ # <-- whenever we use the same 'id', the data get replaced.
        {"id": "A", "values": [0.8, 0.1, 0.1, 0.1], "metadata": {'goal': 'testing'}},
        {"id": "B", "values": [0.5, 0.2, 0.2, 0.2], "metadata": {'goal': 'testing'}},
        {"id": "C", "values": [0.3, 0.9, 0.3, 0.3], "metadata": {'goal': 'testing'}},
        {"id": "D", "values": [0.4, 0.4, 0.7, 0.4], "metadata": {'goal': 'testing'}}
    ]
    )
    '''

def main():
    args = getArgument()


    document = loadPdf(args.pdf_file)
    split_texts = textSplitter(document)    
    vectors = textEmbedding(split_texts)

    upsert("upload-db", args.pinecone_key, vectors, split_texts)

#_____
# main
if __name__ == "__main__":
    main()
