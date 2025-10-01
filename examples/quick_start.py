#!/usr/bin/env python3
"""
Quick Start Example for Speech Emotion Recognition Enhancement

This script demonstrates basic usage of the emotion enhancement models
without requiring the full Jupyter notebook environment.

Requirements:
- Install dependencies: pip install -r requirements.txt
- Setup Kaggle API credentials
- GPU recommended for better performance

Usage:
    python examples/quick_start.py
"""

import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_audio(file_path, target_sr=22050, duration=10):
    """
    Load and preprocess audio file
    
    Args:
        file_path: Path to audio file
        target_sr: Target sampling rate
        duration: Target duration in seconds
    
    Returns:
        Preprocessed audio array
    """
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Pad or truncate to target duration
        target_length = target_sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
            
        return audio
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

def audio_to_melspec(audio, sr=22050, n_mels=128):
    """
    Convert audio to mel-spectrogram
    
    Args:
        audio: Audio time series
        sr: Sampling rate
        n_mels: Number of mel bands
    
    Returns:
        Mel-spectrogram
    """
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=512,
        win_length=1024
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def visualize_spectrogram(mel_spec, title="Mel-Spectrogram", sr=22050):
    """
    Visualize mel-spectrogram
    
    Args:
        mel_spec: Mel-spectrogram to visualize
        title: Plot title
        sr: Sampling rate
    """
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        mel_spec,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def demo_preprocessing():
    """
    Demonstrate audio preprocessing pipeline
    """
    print("=== Speech Emotion Recognition Enhancement - Quick Start ===\n")
    
    # Check for sample audio files
    sample_audio_dir = Path("data/sample_audio")
    
    if not sample_audio_dir.exists():
        print("Sample audio directory not found.")
        print("Please ensure you have audio files in 'data/sample_audio/' directory")
        print("Supported formats: .wav, .mp3, .flac")
        print("\nTo get started:")
        print("1. Create 'data/sample_audio/' directory")
        print("2. Add some emotional speech audio files")
        print("3. Run this script again")
        return
    
    # Find audio files
    audio_files = list(sample_audio_dir.glob("*.wav")) + \
                  list(sample_audio_dir.glob("*.mp3")) + \
                  list(sample_audio_dir.glob("*.flac"))
    
    if not audio_files:
        print("No audio files found in 'data/sample_audio/'")
        print("Please add some audio files and try again.")
        return
    
    print(f"Found {len(audio_files)} audio file(s)")
    
    # Process first audio file as example
    audio_file = audio_files[0]
    print(f"\nProcessing: {audio_file.name}")
    
    # Load and preprocess audio
    audio = load_audio(audio_file)
    if audio is None:
        return
    
    print(f"Audio duration: {len(audio) / 22050:.2f} seconds")
    print(f"Audio shape: {audio.shape}")
    
    # Convert to mel-spectrogram
    mel_spec = audio_to_melspec(audio)
    print(f"Mel-spectrogram shape: {mel_spec.shape}")
    
    # Visualize
    visualize_spectrogram(mel_spec, f"Original - {audio_file.name}")
    
    print("\n=== Next Steps ===")
    print("1. Open the full Jupyter notebook for complete functionality")
    print("2. Train the enhancement models on your dataset")
    print("3. Compare original vs enhanced emotional clarity")
    print("4. Evaluate emotion recognition accuracy improvements")

def create_sample_structure():
    """
    Create sample directory structure
    """
    directories = [
        "data/sample_audio",
        "data/emodb",
        "data/ravdess",
        "models",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    # Create directory structure if it doesn't exist
    create_sample_structure()
    
    # Run demo
    demo_preprocessing()