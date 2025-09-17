# preprocessing/document.py
"""
Document preprocessing utilities.
"""
import PyPDF2
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str or None: Extracted text or None if an error occurred
    """
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            extracted_text = ""
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text
                    
            return extracted_text
            
    except Exception as e:
        logger.error(f"Failed to process document from {pdf_path}: {e}")
        return None