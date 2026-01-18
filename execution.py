import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from preprocessing import preprocess_image
from ocr_module import OCRExtractor
from vlm_module import VLMExtractor
from detection_module import SignatureStampDetector
from field_extractors import FieldExtractor
from validation import Validator
from output_formatter import OutputFormatter


class DocumentAIPipeline:
    """End-to-end pipeline for invoice field extraction"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize pipeline with configuration"""
        self.load_config(config_path)
        
        # Initialize modules
        print("Initializing OCR module...")
        self.ocr_extractor = OCRExtractor()
        
        print("Initializing VLM module...")
        self.vlm_extractor = VLMExtractor()
        
        print("Initializing detection module...")
        self.detector = SignatureStampDetector()
        
        print("Initializing field extractor...")
        self.field_extractor = FieldExtractor(
            dealer_master_path=self.config.get("dealer_master_path", "data/dealer_master.txt"),
            asset_master_path=self.config.get("asset_master_path", "data/asset_master.txt")
        )
        
        print("Initializing validator...")
        self.validator = Validator()
        
        print("Pipeline initialized successfully!")
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "fuzzy_match_threshold": 90,
                "confidence_threshold": 0.5,
                "dealer_master_path": "data/dealer_master.txt",
                "asset_master_path": "data/asset_master.txt"
            }
    
    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single document and extract fields
        
        Args:
            image_path: Path to the invoice image
            
        Returns:
            Dictionary containing extracted fields and metadata
        """
        start_time = time.time()
        
        print(f"\nProcessing: {image_path}")
        
        # Step 1: Preprocess image
        print("  [1/6] Preprocessing image...")
        preprocessed_img = preprocess_image(image_path)
        
        # Step 2: OCR extraction
        print("  [2/6] Extracting text with OCR...")
        ocr_result = self.ocr_extractor.extract(preprocessed_img)
        
        # Step 3: VLM extraction
        print("  [3/6] Analyzing with Vision-Language Model...")
        vlm_result = self.vlm_extractor.extract(preprocessed_img)
        
        # Step 4: Signature and stamp detection
        print("  [4/6] Detecting signatures and stamps...")
        detection_result = self.detector.detect(preprocessed_img)
        
        # Step 5: Field extraction
        print("  [5/6] Extracting and matching fields...")
        fields = self.field_extractor.extract_all_fields(
            ocr_result=ocr_result,
            vlm_result=vlm_result,
            detection_result=detection_result
        )
        
        # Step 6: Validation
        print("  [6/6] Validating and formatting output...")
        validated_fields = self.validator.validate(fields)
        
        # Calculate processing time and cost
        processing_time = time.time() - start_time
        estimated_cost = self.estimate_cost(processing_time)
        
        # Format output
        doc_id = Path(image_path).stem
        output = OutputFormatter.format(
            doc_id=doc_id,
            fields=validated_fields,
            processing_time=processing_time,
            estimated_cost=estimated_cost
        )
        
        print(f"  ✓ Completed in {processing_time:.2f}s (Est. cost: ${estimated_cost:.4f})")
        
        return output
    
    def estimate_cost(self, processing_time: float) -> float:
        """
        Estimate cost per document (all models run locally for free)
        
        Args:
            processing_time: Time taken to process document in seconds
            
        Returns:
            Estimated cost in USD
        """
        # Since we're using local inference (PaddleOCR, Qwen2.5-VL, YOLOv8),
        # the cost is essentially $0, but we can account for electricity/compute
        # Assuming CPU inference at ~0.0003 USD per second (very conservative estimate)
        return processing_time * 0.0003
    
    def process_batch(self, input_dir: str, output_dir: str):
        """
        Process a batch of documents
        
        Args:
            input_dir: Directory containing invoice images
            output_dir: Directory to save output JSON files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.pdf', '.tiff', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        print(f"\nFound {len(image_files)} documents to process")
        
        results = []
        total_time = 0
        total_cost = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n{'='*60}")
            print(f"Document {i}/{len(image_files)}")
            print(f"{'='*60}")
            
            try:
                result = self.process_document(str(image_path))
                results.append(result)
                
                total_time += result['processing_time_seconds']
                total_cost += result['estimated_cost_usd']
                
                # Save individual result
                output_path = Path(output_dir) / f"{result['doc_id']}.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                    
            except Exception as e:
                print(f"  ✗ Error processing {image_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total documents: {len(image_files)}")
        print(f"Successfully processed: {len(results)}")
        print(f"Failed: {len(image_files) - len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per document: {total_time/len(results):.2f}s")
        print(f"Total estimated cost: ${total_cost:.4f}")
        print(f"Average cost per document: ${total_cost/len(results):.6f}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Document AI Invoice Field Extraction")
    parser.add_argument('--input', type=str, required=True, help='Input image file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file or directory')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentAIPipeline(config_path=args.config)
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single document
        result = pipeline.process_document(args.input)
        
        # Save output
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nOutput saved to: {args.output}")
        
    elif os.path.isdir(args.input):
        # Process batch
        pipeline.process_batch(args.input, args.output)
        
    else:
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
