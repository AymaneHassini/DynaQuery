# preprocessing/image.py
"""
Image preprocessing utilities.
"""
from PIL import Image
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

def preprocess_and_check_image(image_url):
    """
    Fetch and preprocess an image from a URL.
    
    Args:
        image_url: URL of the image to process
        
    Returns:
        PIL.Image or None: Processed image or None if an error occurred
    """
    try:
        # Fetch the image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Open and convert the image
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to process image from {image_url}: {e}")
        return None