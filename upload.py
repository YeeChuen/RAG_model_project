# Author(s): Yee
'''
Usage:
    python3 upload.py --pdf_file=Correcting_Diverse_Factual_Errors_paper.pdf
'''
#_____
# imports
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SpacyTextSplitter

#_____
# variables

#_____
# functions
def getArgument():
    parser = argparse.ArgumentParser(description="Upload a PDF file to the RAG model.")
    parser.add_argument("--pdf_file", help=":Path to the PDF file to upload", required=True)
    return parser.parse_args()

def loadPdf(pdf_file):
    loader = PyPDFLoader(pdf_file) # <-- basic pypdf, not going to extract images. (Does not work with my old local Mac)
    pages = loader.load_and_split()
    #print(len(pages)) # <-- development and debug purposes
    #print(pages[0].page_content) # <-- development and debug purposes

    document_list = []

    for i in range(len(pages)):
        document_list.append(pages[i].page_content)

    return "\n".join(document_list)

def textSplitter(document):
    text_splitter = SpacyTextSplitter(chunk_size=200)
    split_texts = text_splitter.split_text(document)
    #print(split_texts[0]) # <-- development and debug purposes
    #print(len(split_texts)) # <-- development and debug purposes

    return split_texts

def main():
    args = getArgument()
    print("Loading PDF file...")

    document = loadPdf(args.pdf_file)
    #print(document) # <-- development and debug purposes.
    textSplitter(document)
    

#_____
# main
if __name__ == "__main__":
    main()
