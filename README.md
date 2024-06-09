# VisionVocabClassifier
15-Scene Image Classification with SIFT and SVM

[中文文档请戳](./README_ZH.md)

This repository contains the Python code for a machine learning project that classifies images from the 15-Scene dataset using Scale-Invariant Feature Transform (SIFT) features and Support Vector Machine (SVM) classifier. The project utilizes OpenCV for image processing, scikit-learn for clustering and classification, and joblib for caching.

## Dataset

The dataset used is the 15-Scene Image Dataset, which can be downloaded from:
[15-Scene Image Dataset](https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12855452/15SceneImageDataset.rar)

## Installation

To run this project, you need Python 3.x and the following packages:
- OpenCV
- NumPy
- scikit-learn
- joblib
- tqdm

You can install the required packages using `pip`:
```bash
pip install numpy opencv-python scikit-learn joblib tqdm
```

## Usage

1. First, clone the repository to your local machine:
   ```bash
   git clone git@github.com:hammershock/VisionVocabClassifier.git
   cd VisionVocabClassifier
   ```

2. Download the 15-Scene Image Dataset and extract it into the project directory under `./15-Scene Image Dataset`.

3. Run the main script:
   ```bash
   python main.py
   ```

This will execute the data loading, feature extraction, training, and evaluation sequence. Results including the model's accuracy and confusion matrix will be displayed in the console.

## Features

- **Data Loading**: Automatically loads and splits the dataset.
- **Feature Extraction**: Extracts SIFT descriptors from images.
- **Clustering**: Uses MiniBatchKMeans for clustering descriptors into visual words.
- **Histogram Generation**: Builds histogram features from clustered descriptors.
- **Classification**: Trains an SVM classifier with RBF kernel.
- **Evaluation**: Computes accuracy and displays a confusion matrix.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
