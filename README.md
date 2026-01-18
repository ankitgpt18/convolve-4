# Document AI for Invoice Field Extraction

Intelligent pipeline for extracting structured fields from tractor loan quotations.

## Overview

Extracts six key fields from invoice documents:
- Dealer Name (fuzzy matched)
- Model Name (exact matched)
- Horse Power
- Asset Cost
- Dealer Signature (detection + bbox)
- Dealer Stamp (detection + bbox)

## Architecture

Hybrid approach combining:
- PaddleOCR for multilingual text extraction
- Qwen2.5-VL-2B for semantic understanding
- YOLOv8-nano for signature/stamp detection
- RapidFuzz for fuzzy matching

## Performance Targets

- Latency: ≤30 seconds per document
- Cost: ≤$0.01 per document
- Accuracy: ≥95% Document Level Accuracy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Single document:
```bash
python execution.py --input invoice.jpg --output result.json
```

Batch processing:
```bash
python execution.py --input dataset/ --output results/
```

## Project Structure

```
├── execution.py          # Main pipeline
├── utils/                # Core modules
│   ├── preprocessing.py
│   ├── ocr_module.py
│   ├── vlm_module.py
│   ├── detection_module.py
│   ├── field_extractors.py
│   ├── validation.py
│   └── output_formatter.py
├── data/                 # Master lists
└── config.json           # Configuration
```

## Configuration

Edit `config.json` to adjust thresholds and model settings.

## Output Format

JSON output includes extracted fields, confidence scores, and processing metadata.

## Requirements

- Python 3.8+
- See requirements.txt for dependencies
