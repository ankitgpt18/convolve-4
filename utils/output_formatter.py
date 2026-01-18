"""
Output formatter for JSON results
"""

from typing import Dict, Any


class OutputFormatter:
    """Format extraction results into required JSON schema"""
    
    @staticmethod
    def format(
        doc_id: str,
        fields: Dict[str, Any],
        processing_time: float,
        estimated_cost: float
    ) -> Dict[str, Any]:
        """
        Format fields into required output schema
        
        Args:
            doc_id: Document identifier
            fields: Validated extracted fields
            processing_time: Time taken to process (seconds)
            estimated_cost: Estimated cost in USD
            
        Returns:
            Formatted output dictionary
        """
        output = {
            "doc_id": doc_id,
            "fields": {
                "dealer_name": fields.get('dealer_name'),
                "model_name": fields.get('model_name'),
                "horse_power": fields.get('horse_power'),
                "asset_cost": fields.get('asset_cost'),
                "dealer_signature": {
                    "present": fields.get('dealer_signature', {}).get('present', False),
                    "bbox": fields.get('dealer_signature', {}).get('bbox')
                },
                "dealer_stamp": {
                    "present": fields.get('dealer_stamp', {}).get('present', False),
                    "bbox": fields.get('dealer_stamp', {}).get('bbox')
                }
            },
            "confidence": {
                "dealer_name": fields.get('dealer_name_confidence', 0.0),
                "model_name": fields.get('model_name_confidence', 0.0),
                "horse_power": fields.get('horse_power_confidence', 0.0),
                "asset_cost": fields.get('asset_cost_confidence', 0.0),
                "dealer_signature": fields.get('dealer_signature', {}).get('confidence', 0.0),
                "dealer_stamp": fields.get('dealer_stamp', {}).get('confidence', 0.0)
            },
            "explanation": {
                "dealer_name": fields.get('dealer_name_explanation', ''),
                "model_name": fields.get('model_name_explanation', ''),
                "horse_power": fields.get('horse_power_explanation', ''),
                "asset_cost": fields.get('asset_cost_explanation', ''),
                "dealer_signature": "Detected" if fields.get('dealer_signature', {}).get('present') else "Not detected",
                "dealer_stamp": "Detected" if fields.get('dealer_stamp', {}).get('present') else "Not detected"
            },
            "processing_time_seconds": round(processing_time, 2),
            "estimated_cost_usd": round(estimated_cost, 6)
        }
        
        return output
