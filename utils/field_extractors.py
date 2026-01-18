import re
from typing import Dict, Any, List, Optional
from rapidfuzz import fuzz, process


class FieldExtractor:
    """Extract and match specific fields from invoice data"""
    
    def __init__(self, dealer_master_path: str, asset_master_path: str):
        """
        Initialize field extractor with master lists
        
        Args:
            dealer_master_path: Path to dealer master list file
            asset_master_path: Path to asset (tractor model) master list file
        """
        self.dealer_master = self._load_master_list(dealer_master_path)
        self.asset_master = self._load_master_list(asset_master_path)
        
        print(f"Loaded {len(self.dealer_master)} dealers and {len(self.asset_master)} models")
    
    def _load_master_list(self, file_path: str) -> List[str]:
        """Load master list from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                items = [line.strip() for line in f if line.strip()]
            return items
        except FileNotFoundError:
            print(f"Warning: Master list not found: {file_path}")
            return []
    
    def extract_all_fields(
        self,
        ocr_result: Dict[str, Any],
        vlm_result: Dict[str, Any],
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract all required fields from combined OCR, VLM, and detection results
        
        Args:
            ocr_result: OCR extraction result
            vlm_result: VLM extraction result
            detection_result: Signature/stamp detection result
            
        Returns:
            Dictionary with all extracted fields
        """
        fields = {}
        
        # Combine text sources
        ocr_text = ocr_result.get('full_text', '')
        
        # Extract Dealer Name
        fields['dealer_name'] = self.extract_dealer_name(ocr_text, vlm_result)
        
        # Extract Model Name
        fields['model_name'] = self.extract_model_name(ocr_text, vlm_result)
        
        # Extract Horse Power
        fields['horse_power'] = self.extract_horse_power(ocr_text, vlm_result)
        
        # Extract Asset Cost
        fields['asset_cost'] = self.extract_asset_cost(ocr_text, vlm_result)
        
        # Extract Signature
        fields['dealer_signature'] = detection_result.get('signature', {
            'present': False,
            'bbox': None
        })
        
        # Extract Stamp
        fields['dealer_stamp'] = detection_result.get('stamp', {
            'present': False,
            'bbox': None
        })
        
        return fields
    
    def extract_dealer_name(self, ocr_text: str, vlm_result: Dict) -> Dict[str, Any]:
        """
        Extract and fuzzy match dealer name
        
        Returns:
            Dict with matched_name, confidence, and explanation
        """
        # Try VLM result first
        vlm_dealer = vlm_result.get('dealer_name')
        
        # Also extract from OCR text (look for common patterns)
        ocr_candidates = self._extract_dealer_candidates(ocr_text)
        
        # Combine candidates
        candidates = []
        if vlm_dealer:
            candidates.append(vlm_dealer)
        candidates.extend(ocr_candidates)
        
        # Fuzzy match against master list
        best_match = None
        best_score = 0
        best_candidate = None
        
        for candidate in candidates:
            if not candidate:
                continue
            
            # Find best match in master list
            result = process.extractOne(
                candidate,
                self.dealer_master,
                scorer=fuzz.ratio
            )
            
            if result and result[1] > best_score:
                best_match = result[0]
                best_score = result[1]
                best_candidate = candidate
        
        # Check if match meets threshold (90%)
        if best_score >= 90:
            return {
                'value': best_match,
                'confidence': best_score / 100,
                'explanation': f"Fuzzy matched '{best_candidate}' to '{best_match}' with {best_score}% similarity"
            }
        else:
            return {
                'value': None,
                'confidence': 0.0,
                'explanation': f"No dealer match found (best: {best_score}%)"
            }
    
    def _extract_dealer_candidates(self, text: str) -> List[str]:
        """Extract potential dealer names from text"""
        candidates = []
        
        # Look for lines with dealer-like keywords
        keywords = ['motors', 'auto', 'tractors', 'pvt', 'ltd', 'limited', 'company', 'dealer']
        
        for line in text.split('\n'):
            line = line.strip()
            if any(kw in line.lower() for kw in keywords):
                # Clean up the line
                cleaned = re.sub(r'[^\w\s]', ' ', line)
                cleaned = ' '.join(cleaned.split())
                if len(cleaned) > 5:
                    candidates.append(cleaned)
        
        return candidates[:5]  # Return top 5 candidates
    
    def extract_model_name(self, ocr_text: str, vlm_result: Dict) -> Dict[str, Any]:
        """
        Extract and exactly match model name
        
        Returns:
            Dict with matched_name, confidence, and explanation
        """
        # Try VLM result first
        vlm_model = vlm_result.get('model_name')
        
        # Extract from OCR (look for model patterns)
        ocr_models = self._extract_model_candidates(ocr_text)
        
        # Combine candidates
        candidates = []
        if vlm_model:
            candidates.append(vlm_model)
        candidates.extend(ocr_models)
        
        # Try exact matching with some OCR error tolerance
        for candidate in candidates:
            if not candidate:
                continue
            
            # Try exact match
            if candidate in self.asset_master:
                return {
                    'value': candidate,
                    'confidence': 1.0,
                    'explanation': f"Exact match found: {candidate}"
                }
            
            # Try fuzzy match (high threshold for "exact"-ish match)
            result = process.extractOne(
                candidate,
                self.asset_master,
                scorer=fuzz.ratio
            )
            
            if result and result[1] >= 95:  # Very high threshold for model names
                return {
                    'value': result[0],
                    'confidence': result[1] / 100,
                    'explanation': f"Matched '{candidate}' to '{result[0]}' ({result[1]}%)"
                }
        
        return {
            'value': None,
            'confidence': 0.0,
            'explanation': "No model match found in asset master"
        }
    
    def _extract_model_candidates(self, text: str) -> List[str]:
        """Extract potential tractor model names"""
        candidates = []
        
        # Common tractor brands
        brands = ['mahindra', 'john deere', 'sonalika', 'tafe', 'new holland', 
                  'kubota', 'massey ferguson', 'farmtrac', 'powertrac']
        
        for line in text.split('\n'):
            line_lower = line.lower()
            if any(brand in line_lower for brand in brands):
                # Extract model info
                cleaned = line.strip()
                candidates.append(cleaned)
        
        return candidates[:10]
    
    def extract_horse_power(self, ocr_text: str, vlm_result: Dict) -> Dict[str, Any]:
        """
        Extract horse power (HP) value
        
        Returns:
            Dict with value (numeric), confidence, and explanation
        """
        # Try VLM result first
        vlm_hp = vlm_result.get('horse_power')
        if vlm_hp is not None:
            return {
                'value': int(vlm_hp),
                'confidence': 0.9,
                'explanation': f"Extracted {int(vlm_hp)} HP from VLM"
            }
        
        # Extract from OCR using regex patterns
        hp_patterns = [
            r'(\d+)\s*HP',
            r'(\d+)\s*hp',
            r'(\d+)\s*horse\s*power',
            r'HP\s*[:\-]?\s*(\d+)',
            r'Power\s*[:\-]?\s*(\d+)',
        ]
        
        for pattern in hp_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                # Take the first match
                hp_value = int(matches[0])
                # Validate range (tractor HP typically 20-150)
                if 20 <= hp_value <= 150:
                    return {
                        'value': hp_value,
                        'confidence': 0.85,
                        'explanation': f"Extracted {hp_value} HP using pattern '{pattern}'"
                    }
        
        return {
            'value': None,
            'confidence': 0.0,
            'explanation': "No HP value found"
        }
    
    def extract_asset_cost(self, ocr_text: str, vlm_result: Dict) -> Dict[str, Any]:
        """
        Extract asset cost (total price)
        
        Returns:
            Dict with value (numeric), confidence, and explanation
        """
        # Try VLM result first
        vlm_cost = vlm_result.get('asset_cost')
        if vlm_cost is not None:
            return {
                'value': float(vlm_cost),
                'confidence': 0.9,
                'explanation': f"Extracted cost ₹{vlm_cost:,.2f} from VLM"
            }
        
        # Extract from OCR using regex patterns
        cost_patterns = [
            r'(?:total|cost|price|amount|value)\s*[:\-]?\s*(?:rs\.?|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:rs\.?|₹)\s*([\d,]+(?:\.\d{2})?)',
            r'([\d,]+(?:\.\d{2})?)\s*(?:rupees|lakhs?)',
        ]
        
        candidates = []
        
        for pattern in cost_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            for match in matches:
                # Clean and convert
                value_str = match.replace(',', '').strip()
                try:
                    value = float(value_str)
                    # Validate range (tractor cost typically 3-15 lakhs)
                    if 300000 <= value <= 1500000:
                        candidates.append((value, pattern))
                except ValueError:
                    continue
        
        if candidates:
            # Take the highest value (likely the total)
            best_value, best_pattern = max(candidates, key=lambda x: x[0])
            return {
                'value': best_value,
                'confidence': 0.85,
                'explanation': f"Extracted cost ₹{best_value:,.2f} using pattern '{best_pattern}'"
            }
        
        return {
            'value': None,
            'confidence': 0.0,
            'explanation': "No valid cost found"
        }
