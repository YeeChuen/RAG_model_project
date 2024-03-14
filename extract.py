# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import PyPDF2
def extract_text_from_pdf(pdf_path):
    # Open the PDF file in binary mode
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object using PdfReader
        reader = PyPDF2.PdfReader(file)

        # Initialize a variable to store the extracted text
        extracted_text = ""

        # Loop through each page in the PDF
        for page in reader.pages:
            # Extract text from the page and add it to the accumulated text
            extracted_text += page.extract_text()

        # Return the extracted text from the PDF
        return extracted_text
