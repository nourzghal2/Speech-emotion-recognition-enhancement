# Speech Emotion Recognition Enhancement

This project implements and extends the methodology from the paper **"A Generation of Enhanced Data by Variational Autoencoders and Diffusion Modeling"** by Young-Jun Kim and Seok-Pil Lee, focusing on enhancing emotional clarity in speech data using deep learning techniques.

## 🎯 Project Overview

The project addresses the challenge of improving emotional clarity in speech data, which is crucial for speech emotion recognition and synthesis applications. We implement diffusion models to enhance emotional features in speech by converting audio to mel-spectrograms and applying generative enhancement techniques.

### Key Features
- **Diffusion Model Implementation**: Core approach from the research paper
- **GAN-based Alternative**: Additional model architecture for comparison
- **Comprehensive Evaluation**: Emotion recognition accuracy assessment
- **Visualization Tools**: Mel-spectrogram analysis and comparison
- **Multi-dataset Support**: EmoDB and RAVDESS datasets

## 📊 Datasets

### EmoDB (Berlin Database of Emotional Speech)
- **Language**: German
- **Speakers**: 10 (5 male, 5 female)
- **Sample Rate**: 16 kHz → normalized to 22,050 Hz
- **Total Files**: 454
- **Emotions**: Neutral, Anger, Fear, Happiness, Sadness, Disgust

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Language**: English
- **Speakers**: 24 (12 male, 12 female)
- **Sample Rate**: 48 kHz → normalized to 22,050 Hz
- **Total Files**: 1,056
- **Emotions**: Neutral, Anger, Fear, Happiness, Sadness, Disgust

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install librosa soundfile
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm
pip install kaggle
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/nourzghal2/Speech-emotion-recognition-enhancement.git
cd Speech-emotion-recognition-enhancement
```

2. Set up Kaggle API credentials for dataset access
3. Open the Jupyter notebook: `enhanced-deep-learning-final (1).ipynb`

## 🏗️ Architecture

### 1. Diffusion Model (Primary Approach)
- **Encoder**: Processes mel-spectrograms into latent representations
- **Diffusion Process**: Adds and removes noise to enhance emotional features
- **Decoder**: Reconstructs enhanced mel-spectrograms
- **U-Net Architecture**: Time-conditional denoising network

### 2. GAN Model (Alternative Approach)
- **Generator**: Creates enhanced emotional features
- **Discriminator**: Ensures realistic emotional expressions
- **Computational Efficiency**: Faster inference compared to diffusion model

### 3. Emotion Recognition Evaluator
- **CNN Architecture**: Evaluates recognition accuracy
- **Multi-class Classification**: 6 emotion categories
- **Performance Metrics**: Weighted Accuracy (WA) and Unweighted Accuracy (UA)

## 📈 Results

Our implementation demonstrates significant improvements in emotion recognition accuracy:

- **Enhanced Data Performance**: Both diffusion and GAN models outperform original data
- **Diffusion Model**: Superior results on subtle emotional expressions
- **GAN Model**: More computationally efficient for real-time applications
- **Visual Enhancement**: Clearer emotional features in enhanced mel-spectrograms

### Key Metrics
- Improved Weighted Accuracy (WA)
- Enhanced Unweighted Accuracy (UA)
- Better confusion matrix scores across all emotion categories

## 🔬 Methodology

1. **Data Preprocessing**
   - Audio normalization to 22.05 kHz
   - Padding to 10-second length
   - Mel-spectrogram conversion

2. **Model Training**
   - Diffusion model with noise scheduling
   - GAN training with adversarial loss
   - Emotion recognition baseline establishment

3. **Enhancement Process**
   - Forward diffusion for feature enhancement
   - Reverse process for reconstruction
   - Quality assessment and validation

4. **Evaluation**
   - Comparative analysis with original data
   - Cross-model performance comparison
   - Statistical significance testing

## 📁 Project Structure

```
Speech-emotion-recognition-enhancement/
├── enhanced-deep-learning-final (1).ipynb  # Main implementation notebook
├── README.md                               # Project documentation
├── data/                                  # Dataset storage (created during runtime)
├── models/                               # Saved model checkpoints
└── results/                             # Output visualizations and metrics
```

## 🛠️ Usage

1. **Run Data Download**: Execute the Kaggle dataset download cells
2. **Preprocessing**: Convert audio files to mel-spectrograms
3. **Model Training**: Train both diffusion and GAN models
4. **Enhancement**: Apply models to generate enhanced data
5. **Evaluation**: Compare recognition accuracy metrics
6. **Visualization**: Analyze enhanced vs. original spectrograms

## 🎯 Applications

- **Speech Emotion Recognition**: Improved accuracy systems
- **Emotional Speech Synthesis**: Clearer emotional expression
- **Human-Computer Interaction**: Better emotional understanding
- **Data Augmentation**: Enhanced training datasets for emotion-aware AI

## 🔮 Future Work

- **Hybrid Models**: Combining diffusion and GAN strengths
- **Multi-language Support**: Extending to additional language datasets
- **Real-time Implementation**: Optimizing for practical applications
- **Perceptual Studies**: Human listener evaluation
- **Fine-tuning**: Domain-specific optimization

## 📝 Citation

If you use this work, please cite the original paper:
```bibtex
@article{kim2023generation,
  title={A Generation of Enhanced Data by Variational Autoencoders and Diffusion Modeling},
  author={Kim, Young-Jun and Lee, Seok-Pil},
  journal={Your Journal},
  year={2023}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

- **Author**: Nour Zghal
- **GitHub**: [@nourzghal2](https://github.com/nourzghal2)
- **Repository**: [Speech-emotion-recognition-enhancement](https://github.com/nourzghal2/Speech-emotion-recognition-enhancement)

## 🙏 Acknowledgments

- Original paper authors: Young-Jun Kim and Seok-Pil Lee
- EmoDB and RAVDESS dataset creators
- PyTorch and Librosa communities for excellent tools

---

⭐ **Star this repository if you find it helpful!**