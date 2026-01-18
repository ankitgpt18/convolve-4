"""
Validation and post-processing logic for extracted fields
"""

from typing import Dict, Any


class Validator:
    """Validate and post-process extracted fields"""
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize validator
        
        Args:
            min_confidence: Minimum confidence threshold for field acceptance
        """
        self.min_confidence = min_confidence
    
    def validate(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all extracted fields
        
        Args:
            fields: Raw extracted fields
            
        Returns:
            Validated and cleaned fields
        """
        validated = {}
        
        # Validate Dealer Name
        dealer = fields.get('dealer_name', {})
        if isinstance(dealer, dict):
            if dealer.get('confidence', 0) >= self.min_confidence:
                validated['dealer_name'] = dealer.get('value')
                validated['dealer_name_confidence'] = dealer.get('confidence')
                validated['dealer_name_explanation'] = dealer.get('explanation')
            else:
                validated['dealer_name'] = None
                validated['dealer_name_confidence'] = dealer.get('confidence', 0.0)
                validated['dealer_name_explanation'] = dealer.get('explanation', 'Low confidence')
        else:
            validated['dealer_name'] = dealer
            validated['dealer_name_confidence'] = 0.5
            validated['dealer_name_explanation'] = 'Direct extraction'
        
        # Validate Model Name
        model = fields.get('model_name', {})
        if isinstance(model, dict):
            if model.get('confidence', 0) >= self.min_confidence:
                validated['model_name'] = model.get('value')
                validated['model_name_confidence'] = model.get('confidence')
                validated['model_name_explanation'] = model.get('explanation')
            else:
                validated['model_name'] = None
                validated['model_name_confidence'] = model.get('confidence', 0.0)
                validated['model_name_explanation'] = model.get('explanation', 'Low confidence')
        else:
            validated['model_name'] = model
            validated['model_name_confidence'] = 0.5
            validated['model_name_explanation'] = 'Direct extraction'
        
        # Validate Horse Power
        hp = fields.get('horse_power', {})
        if isinstance(hp, dict):
            hp_value = hp.get('value')
            if hp_value and self._validate_hp_range(hp_value):
                validated['horse_power'] = int(hp_value)
                validated['horse_power_confidence'] = hp.get('confidence')
                validated['horse_power_explanation'] = hp.get('explanation')
            else:
                validated['horse_power'] = None
                validated['horse_power_confidence'] = hp.get('confidence', 0.0)
                validated['horse_power_explanation'] = hp.get('explanation', 'Invalid range or not found')
        else:
            hp_value = hp
            if hp_value and self._validate_hp_range(hp_value):
                validated['horse_power'] = int(hp_value)
                validated['horse_power_confidence'] = 0.7
                validated['horse_power_explanation'] = 'Direct extraction'
            else:
                validated['horse_power'] = None
                validated['horse_power_confidence'] = 0.0
                validated['horse_power_explanation'] = 'Invalid or not found'
        
        # Validate Asset Cost
        cost = fields.get('asset_cost', {})
        if isinstance(cost, dict):
            cost_value = cost.get('value')
            if cost_value and self._validate_cost_range(cost_value):
                validated['asset_cost'] = float(cost_value)
                validated['asset_cost_confidence'] = cost.get('confidence')
                validated['asset_cost_explanation'] = cost.get('explanation')
            else:
                validated['asset_cost'] = None
                validated['asset_cost_confidence'] = cost.get('confidence', 0.0)
                validated['asset_cost_explanation'] = cost.get('explanation', 'Invalid range or not found')
        else:
            cost_value = cost
            if cost_value and self._validate_cost_range(cost_value):
                validated['asset_cost'] = float(cost_value)
                validated['asset_cost_confidence'] = 0.7
                validated['asset_cost_explanation'] = 'Direct extraction'
            else:
                validated['asset_cost'] = None
                validated['asset_cost_confidence'] = 0.0
                validated['asset_cost_explanation'] = 'Invalid or not found'
        
        # Validate Signature
        signature = fields.get('dealer_signature', {})
        validated['dealer_signature'] = {
            'present': signature.get('present', False),
            'bbox': self._validate_bbox(signature.get('bbox')),
            'confidence': signature.get('confidence', 0.0)
        }
        
        # Validate Stamp
        stamp = fields.get('dealer_stamp', {})
        validated['dealer_stamp'] = {
            'present': stamp.get('present', False),
            'bbox': self._validate_bbox(stamp.get('bbox')),
            'confidence': stamp.get('confidence', 0.0)
        }
        
        return validated
    
    def _validate_hp_range(self, hp_value: float) -> bool:
        """Validate horse power is in reasonable range"""
        try:
            hp = float(hp_value)
            return 15 <= hp <= 200  # Typical tractor HP range
        except:
            return False
    
    def _validate_cost_range(self, cost_value: float) -> bool:
        """Validate cost is in reasonable range"""
        try:
            cost = float(cost_value)
            return 200000 <= cost <= 2000000  # 2L to 20L typical range
        except:
            return False
    
    def _validate_bbox(self, bbox) -> list:
        """Validate and normalize bounding box format"""
        if bbox is None:
            return None
        
        if isinstance(bbox, list) and len(bbox) == 4:
            # Ensure all values are integers and positive
            try:
                x, y, w, h = bbox
                if all(isinstance(v, (int, float)) for v in bbox):
                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        return [int(x), int(y), int(w), int(h)]
            except:
                pass
        
        return None
