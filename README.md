# CNNFeatureExtraction

*** model_state_dict.pth is too large to upload, so if you want to use fine-tuned model, please train the model yourself, resnet50 and CIFAR10 are not big, it won't take much time to fine-tune the model.

writeup file contains all the analysis of this project.

Project Structure:

main.py: This file contains the main functionality of the project. It handles downloading the dataset, initializing the model (either pretrained or fine-tuned), extracting features using different layers, implementing t-SNE for visualization, calculating variances, and displaying results.

utils.py: Contains utility functions and classes required by main.py. This includes functions for dataset handling, model initialization, feature extraction, t-SNE computation, variance calculation, visualization, and a Trainer used to fine-tune the model.

CIFAR10 dataset: The dataset directory containing the CIFAR-10 images.

model_state_dict.pth: Fine-tuned state dictionary of the ResNet-50 model.

Requirements

This project requires commonly used deep learning libraries including numpy, torch, seaborn, matplotlib, scikit-learn, etc.

Usage

To run the project, use the following commands:

  * python main.py pretrained: This uses the pretrained ResNet-50 model to extract features from the last average pooling layer, compute t-SNE visualizations, and quantify variances.
  * python main.py finetuned: This first fine-tunes the ResNet-50 model on CIFAR-10 (time-consuming), then uses the fine-tuned model for feature extraction, visualization, and variance quantification.
  * python main.py finetuned --load_model: Loads a provided state dictionary of the fine-tuned model and uses it for feature extraction, visualization, and variance quantification, saving time compared to fine-tuning.
  * python main.py finetuned --load_model –conv: Loads the fine-tuned model and extracts features from a convolutional layer instead of the last average pooling layer. Useful for comparing feature representations at different network layers.
