# Malware Defender Project - Current Status

## Quick Start

**Active Model**: Random Forest (200 trees, trained on 100K samples)  
**Docker Container**: Running on port **8081**  
**Threshold**: 0.21 (optimized)

### Commands

```bash
# Check container status
docker ps | grep defender-rf-test

# Rebuild and run
docker build -t defender-rf .
docker run -d -p 8081:8080 --memory=1.5g --cpus=1 \
  --env DF_MODEL_THRESH=0.21 \
  --name defender-rf-test defender-rf

# Test on challenge dataset
python -m test \
  -m /home/gamdhameet/challenge/challenge_ds/malware \
  -b /home/gamdhameet/challenge/challenge_ds/goodware \
  --url http://127.0.0.1:8081

# Test single file
curl -X POST http://127.0.0.1:8081/ \
  -H "Content-Type: application/octet-stream" \
  --data-binary @/path/to/file.exe
```

## Current Performance

**Challenge Dataset Results** (116 samples: 95 malware, 21 benign):

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Max Time | 1.685s | < 5s | ✅ PASS |
| Avg Time | 0.390s | - | ✅ |
| FPR | 38.1% | < 1% | ❌ FAIL |
| FNR | 25.3% | < 10% | ❌ FAIL |

**Training Performance** (102K samples):
- Accuracy: 98.2%
- FPR: 0.72% ✓
- FNR: 3.95% ✓

## Project Structure

```
defender/
├── defender/
│   ├── __main__.py              # Entry point (Random Forest only)
│   ├── apps.py                  # Flask app
│   └── models/
│       ├── random_forest_model.py       # Active model
│       ├── random_forest_model.pkl      # 83MB, 200 trees
│       ├── random_forest_scaler.pkl     # Feature scaler
│       └── random_forest_features.pkl   # Feature names (82 features)
├── Dockerfile                   # Production Docker config
├── docker-requirements.txt      # Python dependencies
├── requirements.txt             # Local development
├── README.md                    # Competition instructions
└── FAQ.md                       # Troubleshooting guide
```

## Model Details

**Random Forest Classifier**:
- Trees: 200
- Max depth: 25
- Min samples split: 2
- Min samples leaf: 1
- Max features: sqrt
- Class weight: balanced
- Features: 82 (PE structure + imports + sections)

**Training Data**:
- 100,000 ember samples
- 2,577 DikeDataset samples
- Total: 102,577 samples (68K benign, 34K malware)

## Known Issues

### 1. Distribution Mismatch ⚠️
The challenge dataset contains malware types not well-represented in ember/Dike training data.

**Evidence**:
- Training: 98.2% accuracy, 0.72% FPR, 3.95% FNR
- Challenge: 38.1% FPR, 25.3% FNR
- No threshold can achieve both FPR < 1% AND FNR < 10%

**Impact**: Cannot meet competition requirements without challenge-specific data.

### 2. Threshold Trade-off

| Threshold | FPR | FNR | Best For |
|-----------|-----|-----|----------|
| 0.10 | 69.6% | 12.7% | Low FNR (but terrible FPR) |
| 0.21 | 32.6% | 26.4% | Balance (current) |
| 0.50 | 4.76% | 67.4% | Low FPR (but misses malware) |

## Next Steps to Improve

### Short-term (Can implement now)

1. **Analyze False Negatives**
   ```bash
   # Identify which malware is being missed
   python analyze_failures.py  # Create this script
   ```

2. **Feature Engineering**
   - Add behavioral features
   - Include string analysis
   - Add packer detection
   - Incorporate API call patterns

3. **Collect Challenge-Like Data**
   - Analyze the 24 missed malware samples
   - Find similar samples online
   - Retrain with augmented dataset

### Long-term (Requires more data)

1. **Semi-Supervised Learning**
   - Use challenge dataset for fine-tuning
   - Active learning approach
   - Transfer learning from challenge distribution

2. **Ensemble with Heuristics**
   - Combine ML with signature-based detection
   - Add rule-based fallbacks
   - Multi-stage classification

3. **Advanced Models**
   - Deep learning on raw bytes
   - Graph neural networks for control flow
   - Transformer models on disassembly

## Training Instructions

### Retrain Random Forest

```bash
# With 100K samples (current)
python train_random_forest.py --ember-samples 100000

# With more data (slower, better accuracy)
python train_random_forest.py --ember-samples 500000
```

**Training takes**: ~2 hours for 100K samples

### Data Sources

- **Ember**: `/home/gamdhameet/ember_dataset/ember`
- **DikeDataset**: `/home/gamdhameet/DikeDataset-main/files`
- **Challenge**: `/home/gamdhameet/challenge/challenge_ds`

## Docker Deployment

### Build Image

```bash
docker build -t defender-rf .
```

**Image size**: ~400MB compressed, <1GB uncompressed ✓

### Run with Different Thresholds

```bash
# Default (0.21)
docker run -d -p 8081:8080 --memory=1.5g --cpus=1 \
  --name defender-rf defender-rf

# Custom threshold
docker run -d -p 8081:8080 --memory=1.5g --cpus=1 \
  --env DF_MODEL_THRESH=0.3 \
  --name defender-rf defender-rf
```

### Export for Submission

```bash
# Save image
docker save -o defender-rf.tar defender-rf

# Compress
gzip defender-rf.tar

# Result: defender-rf.tar.gz (should be < 1GB)
```

## Troubleshooting

### Container won't start
```bash
docker logs defender-rf-test
# Check for model loading errors
```

### Port already in use
```bash
# Stop all defender containers
docker stop $(docker ps -a | grep defender | awk '{print $1}')
docker rm $(docker ps -a | grep defender | awk '{print $1}')
```

### Slow inference
- Check if running with CPU limit: `docker stats`
- Feature extraction is the bottleneck (~0.2-0.3s per file)
- Consider caching for repeated files

## Files Overview

### Active Files (Keep These)
- `defender/__main__.py` - Entry point
- `defender/models/random_forest_model.py` - Model implementation
- `defender/models/random_forest_*.pkl` - Trained model files
- `Dockerfile` - Container config
- `docker-requirements.txt` - Dependencies
- `requirements.txt` - Local dev dependencies
- `README.md` - Competition guide
- `FAQ.md` - Help guide

### Training Files (Optional)
- `train_random_forest.py` - Training script
- `defender/models/data_loader.py` - Dataset utilities

### Testing Files (Optional)
- `test/` - Test module
- `compare_models.py` - Model comparison
- `optimize_thresholds.py` - Threshold analysis

## Key Metrics to Track

When retraining or testing:

1. **Validation Metrics** (should be good):
   - Accuracy > 95%
   - FPR < 2%
   - FNR < 5%

2. **Challenge Metrics** (currently failing):
   - FPR < 1% ❌
   - FNR < 10% ❌
   - Max time < 5s ✅

3. **Inference Speed**:
   - Should stay < 2s per file
   - Batch processing: ~2-5 files/sec

## Important Notes

- ⚠️ **Model is production-ready for speed** but **not for accuracy** on challenge data
- ✅ Container meets memory (1.5GB) and CPU (1 core) limits
- ✅ Response time well under 5-second requirement
- ❌ Accuracy metrics fail due to distribution mismatch
- 🔄 **Next priority**: Collect challenge-like malware samples and retrain

## Contact/References

- Competition: https://mlsec.io
- Ember Dataset: https://github.com/endgameinc/ember
- LIEF (PE parsing): https://lief-project.github.io/

---

**Last Updated**: October 14, 2025  
**Model Version**: Random Forest v1.0 (100K samples)  
**Status**: Deployed on port 8081, passing speed tests, failing accuracy tests

