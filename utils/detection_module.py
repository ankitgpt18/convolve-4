import numpy as np
from typing import Dict, List, Tuple, Any
import os

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


class SignatureStampDetector:
    """Detector for signatures and stamps using YOLOv8"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained YOLO model (if None, uses pretrained YOLOv8n)
        """
        self.model = None
        self.model_path = model_path
        
        try:
            if model_path and os.path.exists(model_path):
                # Load custom trained model
                self.model = YOLO(model_path)
                print(f"Loaded custom YOLO model from {model_path}")
            else:
                # Use pretrained YOLOv8-nano as base
                # Note: This will need fine-tuning on signature/stamp data
                self.model = YOLO('yolov8n.pt')
                print("Loaded YOLOv8-nano base model (needs fine-tuning)")
                
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Will use rule-based fallback detection")
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect signatures and stamps in the document
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            # Fallback to rule-based detection
            return self._rule_based_detection(image)
        
        try:
            # Run detection
            results = self.model(image, conf=0.3, verbose=False)
            
            # Parse results
            detections = {
                'signature': {'present': False, 'bbox': None, 'confidence': 0.0},
                'stamp': {'present': False, 'bbox': None, 'confidence': 0.0}
            }
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # Convert to [x, y, width, height]
                    x1, y1, x2, y2 = xyxy
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    
                    # Assuming class 0 = signature, class 1 = stamp
                    # This mapping depends on training data
                    if cls_id == 0:
                        detections['signature'] = {
                            'present': True,
                            'bbox': bbox,
                            'confidence': confidence
                        }
                    elif cls_id == 1:
                        detections['stamp'] = {
                            'present': True,
                            'bbox': bbox,
                            'confidence': confidence
                        }
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return self._rule_based_detection(image)
    
    def _rule_based_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Fallback rule-based detection using image processing
        Looks for ink-dense regions in bottom portion of document
        
        Args:
            image: Input image
            
        Returns:
            Detection results
        """
        import cv2
        
        try:
            h, w = image.shape[:2]
            
            # Focus on bottom 30% of document (typical signature/stamp location)
            bottom_region = image[int(h * 0.7):, :]
            
            # Convert to grayscale
            gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)
            
            # Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = {
                'signature': {'present': False, 'bbox': None, 'confidence': 0.5},
                'stamp': {'present': False, 'bbox': None, 'confidence': 0.5}
            }
            
            # Analyze contours
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (signatures/stamps have specific size ranges)
                if 1000 < area < 50000:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    # Adjust y coordinate to full image
                    y_full = y + int(h * 0.7)
                    
                    # Heuristic: wider objects are likely signatures, squarer ones are stamps
                    aspect_ratio = w_box / h_box if h_box > 0 else 0
                    
                    if aspect_ratio > 1.5:
                        # Likely signature (wider than tall)
                        if not detections['signature']['present']:
                            detections['signature'] = {
                                'present': True,
                                'bbox': [int(x), int(y_full), int(w_box), int(h_box)],
                                'confidence': 0.5
                            }
                    elif 0.7 < aspect_ratio < 1.3:
                        # Likely stamp (more square-ish)
                        if not detections['stamp']['present']:
                            detections['stamp'] = {
                                'present': True,
                                'bbox': [int(x), int(y_full), int(w_box), int(h_box)],
                                'confidence': 0.5
                            }
            
            return detections
            
        except Exception as e:
            print(f"Rule-based detection error: {e}")
            return {
                'signature': {'present': False, 'bbox': None, 'confidence': 0.0},
                'stamp': {'present': False, 'bbox': None, 'confidence': 0.0}
            }
    
    def train(self, data_yaml_path: str, epochs: int = 50):
        """
        Train YOLO model on signature/stamp dataset
        
        Args:
            data_yaml_path: Path to YOLO dataset configuration
            epochs: Number of training epochs
        """
        if self.model is None:
            self.model = YOLO('yolov8n.pt')
        
        # Train model
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=8,
            device='cpu',
            project='signature_stamp_detector',
            name='yolov8n_custom'
        )
        
        print(f"Training complete! Model saved.")
        
        return results


import os
