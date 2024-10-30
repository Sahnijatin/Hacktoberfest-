import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
import json
import tempfile
import os
import re
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class OCRProcessor:
    def __init__(self):
        # Update Tesseract configuration to use Debian's default path
        self.tesseract_cmd = '/usr/bin/tesseract'
        self.poppler_path = '/usr/bin'
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        
        # Verify Tesseract installation
        self._verify_tesseract()

    def _verify_tesseract(self):
        """Verify Tesseract installation and available languages."""
        try:
            languages = pytesseract.get_languages()
            logger.info(f"Available Tesseract languages: {languages}")
            if 'eng' not in languages:
                raise RuntimeError("English language pack not found in Tesseract")
        except Exception as e:
            logger.error(f"Tesseract verification failed: {e}")
            raise RuntimeError(f"Tesseract verification failed: {str(e)}")

    def process_pdf(self, pdf_path: str) -> Optional[str]:
        """Process PDF and extract text using OCR."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            logger.info(f"Tesseract CMD: {self.tesseract_cmd}")
            logger.info(f"TESSDATA_PREFIX: {os.getenv('TESSDATA_PREFIX')}")
            
            images = convert_from_path(
                pdf_path,
                poppler_path=self.poppler_path,
                grayscale=True,
                dpi=300
            )
            
            ocr_text = []
            for i, img in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)}")
                text = pytesseract.image_to_string(
                    img,
                    lang='eng',
                    config='--psm 1 --oem 3'
                )
                ocr_text.append(text)
            
            return " ".join(ocr_text)
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            st.error(f"OCR processing failed: {str(e)}")
            return None

class FileHandler:
    @staticmethod
    def save_uploaded_file(uploaded_file) -> Optional[str]:
        """Save uploaded file and return the temporary path."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                return tmp_file.name
        except Exception as e:
            logger.error(f"File save failed: {e}")
            st.error(f"Failed to save file: {str(e)}")
            return None

class JSONProcessor:
    @staticmethod
    def sanitize_response(response: str) -> Optional[Dict[str, Any]]:
        """Sanitize and parse JSON response."""
        try:
            # Remove code blocks and clean whitespace
            cleaned_response = re.sub(r'```.*?```', '', response, flags=re.DOTALL).strip()
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            st.error(f"Failed to parse JSON: {str(e)}")
            return None

class LLMProcessor:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=self.openai_api_key
        )
        
        self.template = """
        You are given a text extracted from a PDF document. Your task is to:
        1. Infer all relevant fields and data from the text
        2. Structure it in a JSON format without manual field mapping
        3. Extract only important, factual data points
        4. Avoid hallucination - only include fields that exist in the PDF
        
        Text:
        {document_text}

        Return the data in clean JSON format.
        """
        
        self.prompt = PromptTemplate(
            input_variables=["document_text"],
            template=self.template
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_json(self, document_text: str) -> str:
        """Generate JSON response using LLM."""
        try:
            return self.chain.run(document_text=document_text)
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            st.error(f"Failed to generate JSON: {str(e)}")
            return None

class StreamlitUI:
    @staticmethod
    def setup():
        """Set up the Streamlit user interface."""
        st.set_page_config(
            page_title="PDF OCR and Data Extraction",
            page_icon="📄",
            layout="centered",
            initial_sidebar_state="auto"
        )
        st.header("PDF OCR and Data Extraction 📄")

    @staticmethod
    def show_error(error: str):
        """Display error message."""
        st.error(error)

    @staticmethod
    def show_success(message: str):
        """Display success message."""
        st.success(message)

    @staticmethod
    def show_json(data: Dict[str, Any]):
        """Display JSON data."""
        st.json(data)

def main():
    try:
        # Initialize components
        StreamlitUI.setup()
        
        # Initialize OCR processor with verification
        try:
            ocr_processor = OCRProcessor()
            logger.info("OCR processor initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize OCR processor: {str(e)}")
            st.error("Please check if Tesseract is properly installed and configured")
            return
        
        llm_processor = LLMProcessor()
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if st.button("Extract and Generate JSON") and uploaded_file:
            with st.spinner("Processing document..."):
                # Process file
                pdf_path = FileHandler.save_uploaded_file(uploaded_file)
                if not pdf_path:
                    return
                
                try:
                    # Extract text
                    ocr_text = ocr_processor.process_pdf(pdf_path)
                    if not ocr_text:
                        return
                    
                    StreamlitUI.show_success("OCR completed successfully")
                    logger.info("OCR text extracted successfully")
                    
                    # Generate and process JSON
                    result = llm_processor.generate_json(ocr_text)
                    sanitized_json = JSONProcessor.sanitize_response(result)
                    
                    if sanitized_json:
                        StreamlitUI.show_json(sanitized_json)
                    
                finally:
                    # Cleanup
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                        logger.info(f"Cleaned up temporary file: {pdf_path}")
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        StreamlitUI.show_error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
