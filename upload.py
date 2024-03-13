# Author(s): Yee
'''
Usage:
    python upload.py --pdf_file=Correcting_Diverse_Factual_Errors_paper.pdf
'''
#_____
# imports
import argparse
from langchain_community.document_loaders import PyPDFLoader

#_____
# variables

#_____
# functions
def getArgument():
    parser = argparse.ArgumentParser(description="Upload a PDF file to the RAG model.")
    parser.add_argument("--pdf_file", help=":Path to the PDF file to upload", required=True)
    return parser.parse_args()

def main():
    args = getArgument()
    print("Loading PDF file...")
    
    loader = PyPDFLoader(args.pdf_file, extract_images=True)
    pages = loader.load()
    print(pages[0])

#_____
# main
if __name__ == "__main__":
    main()
