import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def preprocess_image(image_path: str) -> np.ndarray:
    if isinstance(image_path, str):
        img = cv2.imread(str(image_path))
    else:
        img = image_path
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply preprocessing pipeline
    img = deskew(img)
    img = denoise(img)
    img = enhance_contrast(img)
    img = binarize_adaptive(img)
    
    return img


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Automatically deskew the image using Hough transform
    
    Args:
        image: Input image
        
    Returns:
        Deskewed image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None:
        return image
    
    # Calculate average angle
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return image
    
    median_angle = np.median(angles)
    
    # Rotate image if skew is significant
    if abs(median_angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    return image


def denoise(image: np.ndarray) -> np.ndarray:
    """
    Remove noise from the image
    
    Args:
        image: Input image
        
    Returns:
        Denoised image
    """
    # Use Non-local Means Denoising for color images
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE
    
    Args:
        image: Input image
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def binarize_adaptive(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding for better text extraction
    
    Args:
        image: Input image
        
    Returns:
        Binarized image (kept in RGB for consistency)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to RGB
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    return binary_rgb


def resize_if_large(image: np.ndarray, max_dimension: int = 2048) -> np.ndarray:
    """
    Resize image if too large to speed up processing
    
    Args:
        image: Input image
        max_dimension: Maximum width or height
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    return image
