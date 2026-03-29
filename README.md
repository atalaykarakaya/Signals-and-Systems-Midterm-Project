# Speech Analysis & Gender Classification (COE216) 🎙️

This repository contains the complete source code, metadata, and methodology for the **COE216 Signals and Systems** Midterm Project. The goal is to perform gender classification (Male, Female, Child) using time-domain audio features.

## 📂 Repository Structure
- `speech_analyzer.py`: The main Python application featuring a modern CustomTkinter GUI for single-file and batch analysis.
- [cite_start]`Master_Metadata_Cleaned.csv`: Cleaned metadata file containing ground truth labels (Gender, Age, Emotion) for 300+ recordings[cite: 72].
- [cite_start]`Dataset/`: Directory containing labeled .wav files organized by group folders[cite: 16].

## 🚀 Key Features
- **Time-Domain Analysis**: Automated extraction of Short-Term Energy (STE) and Zero Crossing Rate (ZCR) to identify voiced regions.
- **Pitch Estimation**: Fundamental Frequency ($F_0$) detection using the **Autocorrelation Method**.
- **Live Comparison**: Real-time side-by-side visualization of Autocorrelation vs. FFT Magnitude Spectrum.
- **Batch Processing**: Tools to analyze large datasets and generate success metrics (Accuracy, Confusion Matrix).

## 🛠️ Setup & Installation
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/[REPO_NAME].git
2. Install dependencies:
   pip install numpy pandas librosa scipy matplotlib customtkinter

3. Run the application:
   python speech_analyzer.py

