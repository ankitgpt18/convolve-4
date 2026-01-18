# Quick Start Guide

## Setup (First Time Only)

### 1. Install Dependencies
```bash
cd "C:\Users\ankit\OneDrive\Desktop\Convolve 4.0"
python install_dependencies.py
```

### 2. Verify Setup
```bash
python verify_setup.py
```

### 3. Add Dataset
- Download dataset from WeTransfer link
- Extract to a folder (e.g., `dataset/`)
- Update master lists in `data/dealer_master.txt` and `data/asset_master.txt`

## Running the Pipeline

### Single Document
```bash
python execution.py --input dataset/invoice_001.jpg --output results/invoice_001.json
```

### Batch Processing
```bash
python execution.py --input dataset/ --output results/
```

## Expected Performance

- **Latency**: ~25-30 seconds per document
- **Cost**: ~$0.007 per document (local CPU inference)
- **Accuracy**: Target ≥95% Document Level Accuracy

## Workflow Once Dataset is Available

1. **Extract master lists from dataset**:
   - Run through a few samples manually
   - Extract unique dealer names → `data/dealer_master.txt`
   - Extract unique model names → `data/asset_master.txt`

2. **Test on sample documents**:
   ```bash
   python execution.py --input dataset/sample1.jpg --output test_output/sample1.json
   ```

3. **Manually verify results** and adjust:
   - Check fuzzy matching threshold in `config.json`
   - Add missing entries to master lists
   - Test again

4. **Run full batch**:
   ```bash
   python execution.py --input dataset/ --output final_results/
   ```

5. **Generate submission ZIP**:
   - Include: `execution.py`, `utils/`, `config.json`, `requirements.txt`, `README.md`, `sample_output/`, `data/`
   - Exclude: `dataset/`, `results/`, model weights, `__pycache__/`

## Troubleshooting

### Models downloading on first run
This is normal. PaddleOCR, Qwen2.5-VL, and YOLO will download weights (~2-3 GB total) on first execution.

### Low accuracy
- Check and update master lists
- Adjust fuzzy matching threshold in config.json
- Review error_analysis.txt for patterns

### High latency (>30s)
- Reduce VLM model size or skip VLM (use OCR only mode)
- Disable preprocessing steps in config.json
- Use GPU if available (set use_gpu: true in config.json)

## Contact
For issues during hackathon, check:
- README.md for detailed architecture
- implementation_plan.md in brain folder for design decisions
