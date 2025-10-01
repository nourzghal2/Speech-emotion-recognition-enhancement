# Setup Guide

This guide provides detailed instructions for setting up the Speech Emotion Recognition Enhancement project.

## Prerequisites

### System Requirements
- Python 3.7 or higher
- GPU recommended (CUDA-compatible) for faster training
- At least 8GB RAM
- 10GB free disk space for datasets

### Software Dependencies
- Git
- Jupyter Notebook or JupyterLab
- Kaggle account for dataset access

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/nourzghal2/Speech-emotion-recognition-enhancement.git
cd Speech-emotion-recognition-enhancement
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n speech-emotion python=3.8
conda activate speech-emotion

# Using venv
python -m venv speech-emotion
# On Windows:
speech-emotion\Scripts\activate
# On macOS/Linux:
source speech-emotion/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Kaggle API

#### Option A: Using Kaggle CLI
1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account settings → API → Create New API Token
3. Download `kaggle.json` file
4. Place it in the appropriate location:
   - Windows: `C:\Users\{username}\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`
5. Set permissions (macOS/Linux only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

#### Option B: Manual Dataset Download
If you prefer not to use Kaggle API:
1. Download EmoDB dataset manually
2. Download RAVDESS dataset manually
3. Place them in the `data/` directory (will be created automatically)

### 5. Launch Jupyter Notebook
```bash
jupyter notebook
# or
jupyter lab
```

### 6. Open the Main Notebook
Navigate to and open: `enhanced-deep-learning-final (1).ipynb`

## Dataset Setup

### Automatic Download (Recommended)
The notebook includes cells that automatically download datasets using Kaggle API.

### Manual Setup
If downloading manually:
1. Create directory structure:
   ```
   data/
   ├── emodb/
   └── ravdess/
   ```
2. Extract datasets to respective folders
3. Update file paths in the notebook if necessary

## GPU Setup (Optional but Recommended)

### CUDA Installation
1. Install NVIDIA drivers
2. Install CUDA toolkit (version compatible with PyTorch)
3. Install cuDNN

### Verify GPU Access
Run this in a Python cell:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Common Issues

#### 1. Kaggle API Authentication Error
- Ensure `kaggle.json` is in the correct location
- Check file permissions (should be 600 on Unix systems)
- Verify your Kaggle account has API access enabled

#### 2. CUDA/GPU Issues
- Update NVIDIA drivers
- Reinstall PyTorch with correct CUDA version:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

#### 3. Memory Issues
- Reduce batch size in the notebook
- Use CPU instead of GPU if necessary
- Close other applications to free up RAM

#### 4. Audio Processing Errors
- Ensure librosa and soundfile are properly installed
- Check audio file formats are supported
- Verify sample rates are consistent

#### 5. Import Errors
- Ensure all requirements are installed: `pip install -r requirements.txt`
- Check Python version compatibility
- Try reinstalling problematic packages

### Getting Help
1. Check the [Issues](https://github.com/nourzghal2/Speech-emotion-recognition-enhancement/issues) page
2. Create a new issue with:
   - Error message
   - System information
   - Steps to reproduce
3. Join discussions in the repository

## Performance Tips

1. **Use GPU**: Significantly faster training and inference
2. **Batch Processing**: Process multiple files simultaneously
3. **Data Caching**: Enable caching for repeated data loading
4. **Memory Management**: Clear unused variables with `del` and `gc.collect()`

## Next Steps

After successful setup:
1. Run through the notebook cells sequentially
2. Experiment with different parameters
3. Try your own audio files
4. Contribute improvements back to the project

Happy coding! 🎉