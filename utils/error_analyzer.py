"""
Error analysis module for categorizing and analyzing failures
"""

from typing import Dict, List, Any
from collections import defaultdict
import json


class ErrorAnalyzer:
    """Analyze and categorize extraction errors"""
    
    def __init__(self):
        """Initialize error analyzer"""
        self.error_categories = {
            'ocr_errors': [],
            'extraction_errors': [],
            'matching_errors': [],
            'layout_errors': [],
            'detection_errors': []
        }
    
    def analyze_result(self, result: Dict[str, Any], ground_truth: Dict[str, Any] = None):
        """
        Analyze a single extraction result
        
        Args:
            result: Extraction result from pipeline
            ground_truth: Optional ground truth for comparison
        """
        doc_id = result.get('doc_id')
        fields = result.get('fields', {})
        confidences = result.get('confidence', {})
        
        # Check for missing fields
        for field_name in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            if fields.get(field_name) is None:
                self._log_error(
                    'extraction_errors',
                    doc_id,
                    field_name,
                    'Field not extracted',
                    confidences.get(field_name, 0.0)
                )
        
        # Check for low confidence detections
        for field_name, confidence in confidences.items():
            if confidence < 0.6:
                self._log_error(
                    'extraction_errors',
                    doc_id,
                    field_name,
                    f'Low confidence extraction ({confidence:.2f})',
                    confidence
                )
        
        # Check signature/stamp detection
        if not fields.get('dealer_signature', {}).get('present'):
            self._log_error(
                'detection_errors',
                doc_id,
                'dealer_signature',
                'Signature not detected',
                confidences.get('dealer_signature', 0.0)
            )
        
        if not fields.get('dealer_stamp', {}).get('present'):
            self._log_error(
                'detection_errors',
                doc_id,
                'dealer_stamp',
                'Stamp not detected',
                confidences.get('dealer_stamp', 0.0)
            )
        
        # If ground truth is available, compare
        if ground_truth:
            self._compare_with_ground_truth(result, ground_truth)
    
    def _log_error(self, category: str, doc_id: str, field: str, reason: str, confidence: float):
        """Log an error"""
        self.error_categories[category].append({
            'doc_id': doc_id,
            'field': field,
            'reason': reason,
            'confidence': confidence
        })
    
    def _compare_with_ground_truth(self, result: Dict, ground_truth: Dict):
        """Compare result with ground truth"""
        doc_id = result.get('doc_id')
        result_fields = result.get('fields', {})
        gt_fields = ground_truth.get('fields', {})
        
        # Check each field
        for field_name in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            result_value = result_fields.get(field_name)
            gt_value = gt_fields.get(field_name)
            
            if result_value != gt_value:
                self._log_error(
                    'matching_errors',
                    doc_id,
                    field_name,
                    f'Mismatch: got "{result_value}", expected "{gt_value}"',
                    result.get('confidence', {}).get(field_name, 0.0)
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get error analysis summary
        
        Returns:
            Summary statistics by error category
        """
        summary = {}
        
        for category, errors in self.error_categories.items():
            summary[category] = {
                'count': len(errors),
                'errors': errors
            }
        
        total_errors = sum(len(errors) for errors in self.error_categories.values())
        summary['total_errors'] = total_errors
        
        return summary
    
    def generate_report(self, output_path: str):
        """Generate error analysis report"""
        summary = self.get_summary()
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Errors: {summary['total_errors']}\n\n")
            
            for category, data in summary.items():
                if category == 'total_errors':
                    continue
                
                count = data['count']
                if count == 0:
                    continue
                
                f.write(f"\n{category.upper().replace('_', ' ')}: {count}\n")
                f.write("-" * 40 + "\n")
                
                for error in data['errors'][:10]:  # Show first 10
                    f.write(f"  Doc: {error['doc_id']}\n")
                    f.write(f"  Field: {error['field']}\n")
                    f.write(f"  Reason: {error['reason']}\n")
                    f.write(f"  Confidence: {error['confidence']:.2f}\n\n")
        
        print(f"Error analysis report saved to: {output_path}")
