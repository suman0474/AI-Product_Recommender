"""
File Extraction Utilities
Extracts text from various file formats: PDF, DOCX, DOC, TXT, Images (OCR)
"""

import logging
import io
from typing import Optional, Dict, Any
from PIL import Image
import fitz  # PyMuPDF for PDF
from docx import Document  # python-docx for DOCX

# Initialize logger first
logger = logging.getLogger(__name__)

# Make pytesseract optional (only needed for image OCR)
try:
    import pytesseract
    import os
    
    # Check if Tesseract executable exists
    tesseract_cmd = None
    
    # Common installation paths for different platforms
    windows_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
    ]
    
    # Linux/Railway paths (installed via apt)
    linux_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
    ]
    
    # Check environment variable first
    if os.getenv('TESSERACT_CMD'):
        tesseract_cmd = os.getenv('TESSERACT_CMD')
        logger.info(f"Using Tesseract from TESSERACT_CMD env var: {tesseract_cmd}")
    else:
        # Determine platform and check appropriate paths
        import platform
        system = platform.system()
        
        if system == 'Windows':
            paths_to_check = windows_paths
        else:  # Linux, Darwin (macOS), etc.
            paths_to_check = linux_paths
        
        logger.info(f"Detected platform: {system}, checking paths: {paths_to_check}")
        
        # Check common installation paths
        for path in paths_to_check:
            if os.path.isfile(path):
                tesseract_cmd = path
                logger.info(f"Found Tesseract at: {path}")
                break
    
    # Set tesseract command path if found
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        TESSERACT_AVAILABLE = True
        logger.info(f"✓ Tesseract OCR configured at: {tesseract_cmd}")
    else:
        TESSERACT_AVAILABLE = False
        logger.warning(
            "⚠ Tesseract executable not found. "
            "For Railway deployment, ensure nixpacks.toml includes tesseract-ocr. "
            "For local Windows, install from: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("⚠ pytesseract not installed - image OCR will not be available")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF file
    
    Args:
        file_bytes: PDF file as bytes
        
    Returns:
        Extracted text
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        doc.close()
        
        extracted_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(extracted_text)} characters from PDF ({len(text_parts)} pages)")
        return extracted_text
        
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from DOCX file
    
    Args:
        file_bytes: DOCX file as bytes
        
    Returns:
        Extracted text
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
        text_parts = []
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                if row_text.strip():
                    text_parts.append(row_text)
        
        extracted_text = "\n".join(text_parts)
        logger.info(f"Extracted {len(extracted_text)} characters from DOCX")
        return extracted_text
        
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        return ""


def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Extract text from image using OCR (Tesseract)
    
    Args:
        file_bytes: Image file as bytes
        
    Returns:
        Extracted text via OCR
    """
    if not TESSERACT_AVAILABLE:
        error_msg = (
            "Tesseract OCR is not installed or not found in PATH. "
            "To enable image text extraction:\n"
            "1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2. Install it (default path: C:\\Program Files\\Tesseract-OCR)\n"
            "3. Add Tesseract to your PATH or set TESSERACT_CMD environment variable\n"
            "4. Restart the backend server"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    try:
        # Open and validate image
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB if necessary (some formats like RGBA or P mode can cause issues)
        if image.mode not in ('RGB', 'L'):
            logger.info(f"Converting image from {image.mode} to RGB for OCR")
            image = image.convert('RGB')
        
        # Perform OCR with optimized config
        # --psm 6: Assume a single uniform block of text
        # --oem 3: Use both legacy and LSTM OCR engines
        text = pytesseract.image_to_string(image, config='--psm 6 --oem 3')
        
        extracted_text = text.strip()
        logger.info(f"Extracted {len(extracted_text)} characters from image via OCR")
        
        if not extracted_text:
            logger.warning("OCR completed but no text was found in the image")
            raise RuntimeError("No text found in image. The image may be blank, too low quality, or contain only graphics.")
        
        return extracted_text
        
    except pytesseract.TesseractNotFoundError as e:
        error_msg = (
            f"Tesseract executable not found: {e}\n"
            "Please install Tesseract OCR:\n"
            "Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Linux: sudo apt-get install tesseract-ocr\n"
            "Mac: brew install tesseract"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except RuntimeError:
        # Re-raise RuntimeError as-is (already formatted)
        raise
    except Exception as e:
        logger.error(f"Failed to extract text from image: {e}")
        logger.exception(e)
        raise RuntimeError(f"Image OCR failed: {str(e)}")


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Extract text from plain text file
    
    Args:
        file_bytes: Text file as bytes
        
    Returns:
        Decoded text
    """
    try:
        # Try UTF-8 first
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1
            text = file_bytes.decode('latin-1')
        
        logger.info(f"Extracted {len(text)} characters from text file")
        return text.strip()
        
    except Exception as e:
        logger.error(f"Failed to extract text from text file: {e}")
        return ""


def extract_text_from_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract text from uploaded file based on file type
    
    Args:
        file_bytes: File content as bytes
        filename: Original filename with extension
        
    Returns:
        Dict with extracted text and metadata
    """
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    logger.info(f"Extracting text from file: {filename} (type: {file_ext})")
    
    extracted_text = ""
    file_type = ""
    
    # PDF files
    if file_ext == 'pdf':
        file_type = "PDF"
        extracted_text = extract_text_from_pdf(file_bytes)
    
    # DOCX files
    elif file_ext == 'docx':
        file_type = "DOCX"
        extracted_text = extract_text_from_docx(file_bytes)
    
    # DOC files (older Word format - requires conversion or different library)
    elif file_ext == 'doc':
        file_type = "DOC"
        logger.warning("DOC format not fully supported. Please convert to DOCX or PDF.")
        extracted_text = ""
    
    # Text files
    elif file_ext in ['txt', 'text', 'log']:
        file_type = "TEXT"
        extracted_text = extract_text_from_txt(file_bytes)
    
    # Image files (OCR)
    elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']:
        file_type = "IMAGE"
        try:
            extracted_text = extract_text_from_image(file_bytes)
        except RuntimeError as e:
            logger.warning(f"Image OCR failed for {filename}: {e}")
            return {
                "filename": filename,
                "file_type": file_type,
                "extracted_text": "",
                "character_count": 0,
                "success": False,
                "error": str(e)
            }
    
    else:
        logger.warning(f"Unsupported file type: {file_ext}")
        file_type = "UNSUPPORTED"
    
    return {
        "filename": filename,
        "file_type": file_type,
        "extracted_text": extracted_text,
        "character_count": len(extracted_text),
        "success": len(extracted_text) > 0
    }
