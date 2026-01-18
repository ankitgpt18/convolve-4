import numpy as np
from typing import List, Dict, Tuple, Any
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Warning: PaddleOCR not installed. Run: pip install paddleocr")


class OCRExtractor:
    
    def __init__(self, languages: List[str] = ['en', 'hi', 'te']):
            languages: List of language codes (en=English, hi=Hindi, te=Telugu/South Indian)
        """
        try:
            # Initialize PaddleOCR with multilingual support
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Enable angle classification
                lang='en',  # Primary language
                use_gpu=False,  # Use CPU for cost efficiency
                show_log=False
            )
        except Exception as e:
            print(f"Error initializing PaddleOCR: {e}")
            self.ocr = None
    
    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text from image with bounding boxes
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            Dictionary containing extracted text, bounding boxes, and confidence scores
        """
        if self.ocr is None:
            # Fallback: return empty result
            return {
                'text_lines': [],
                'full_text': '',
                'bboxes': [],
                'confidences': []
            }
        
        try:
            # Run OCR
            result = self.ocr.ocr(image, cls=True)
            
            if result is None or len(result) == 0:
                return {
                    'text_lines': [],
                    'full_text': '',
                    'bboxes': [],
                    'confidences': []
                }
            
            # Parse results
            text_lines = []
            bboxes = []
            confidences = []
            
            for line in result[0]:
                bbox = line[0]  # Bounding box coordinates
                text_info = line[1]  # (text, confidence)
                text = text_info[0]
                confidence = text_info[1]
                
                text_lines.append(text)
                bboxes.append(bbox)
                confidences.append(confidence)
            
            # Combine all text
            full_text = '\n'.join(text_lines)
            
            return {
                'text_lines': text_lines,
                'full_text': full_text,
                'bboxes': bboxes,
                'confidences': confidences,
                'num_lines': len(text_lines),
                'avg_confidence': np.mean(confidences) if confidences else 0.0
            }
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return {
                'text_lines': [],
                'full_text': '',
                'bboxes': [],
                'confidences': []
            }
    
    def extract_regions(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Extract text from specific regions of the image
        
        Args:
            image: Input image
            regions: List of (x, y, width, height) tuples
            
        Returns:
            List of extracted text strings for each region
        """
        results = []
        
        for x, y, w, h in regions:
            # Crop region
            roi = image[y:y+h, x:x+w]
            
            # Extract text from region
            result = self.extract(roi)
            results.append(result['full_text'])
        
        return results
