import torch
import numpy as np
from typing import Dict, Any
from PIL import Image

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("Warning: transformers not installed. Run: pip install transformers")


class VLMExtractor:
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initialize VLM extractor
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        try:
            print(f"Loading VLM model: {model_name}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model with quantization for CPU efficiency
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print("VLM model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading VLM model: {e}")
            print("Will use OCR-only fallback mode")
    
    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract fields from invoice using vision-language understanding
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing extracted fields
        """
        if self.model is None or self.processor is None:
            return self._fallback_extraction()
        
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Create prompt for field extraction
            prompt = self._create_extraction_prompt()
            
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt"
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode response
            response = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]
            
            # Parse response to extract fields
            fields = self._parse_vlm_response(response)
            
            return fields
            
        except Exception as e:
            print(f"VLM extraction error: {e}")
            return self._fallback_extraction()
    
    def _create_extraction_prompt(self) -> str:
        """Create a structured prompt for field extraction"""
        prompt = """You are analyzing a tractor loan quotation/invoice document. Extract the following information:

1. Dealer Name: The name of the dealer/company issuing this quotation
2. Model Name: The tractor model name (e.g., "Mahindra 475 DI", "John Deere 5045D")
3. Horse Power (HP): The engine power in HP (extract only the number)
4. Asset Cost: The total cost/price of the tractor (extract only the number, no currency symbols)

Respond in this exact format:
DEALER_NAME: <dealer name>
MODEL_NAME: <model name>
HORSE_POWER: <numeric value>
ASSET_COST: <numeric value>

If any field is not found, write "NOT_FOUND" for that field."""

        return prompt
    
    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the VLM response to extract structured fields
        
        Args:
            response: Raw response from VLM
            
        Returns:
            Dictionary with extracted fields
        """
        fields = {
            'dealer_name': None,
            'model_name': None,
            'horse_power': None,
            'asset_cost': None,
            'vlm_raw_response': response
        }
        
        # Parse response line by line
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith('DEALER_NAME:'):
                value = line.split(':', 1)[1].strip()
                if value != 'NOT_FOUND':
                    fields['dealer_name'] = value
            
            elif line.startswith('MODEL_NAME:'):
                value = line.split(':', 1)[1].strip()
                if value != 'NOT_FOUND':
                    fields['model_name'] = value
            
            elif line.startswith('HORSE_POWER:'):
                value = line.split(':', 1)[1].strip()
                if value != 'NOT_FOUND':
                    try:
                        fields['horse_power'] = float(value)
                    except:
                        pass
            
            elif line.startswith('ASSET_COST:'):
                value = line.split(':', 1)[1].strip()
                if value != 'NOT_FOUND':
                    try:
                        # Remove commas and parse
                        value = value.replace(',', '')
                        fields['asset_cost'] = float(value)
                    except:
                        pass
        
        return fields
    
    def _fallback_extraction(self) -> Dict[str, Any]:
        """Fallback when VLM is not available"""
        return {
            'dealer_name': None,
            'model_name': None,
            'horse_power': None,
            'asset_cost': None,
            'vlm_available': False
        }
