Cancer Image Classification using ResNet50 and Binary Artificial Algae Algorithm
This project presents an image classification pipeline for identifying colorectal cancer from histopathology images. It integrates deep feature extraction using a pre-trained ResNet50 model with feature selection via a Binary Artificial Algae Algorithm (Binary AAA), followed by classification through an ensemble Voting Classifier.

Dataset
Name: NCT-CRC-HE-100K (Colorectal Cancer Histology Dataset)

Classes: Cancerous and Normal

Format: .tif, .jpg, .png

Sample Limit: Up to 400 images per class used to ensure balanced training and manageable compute time

Pipeline Overview
1. Feature Extraction
Utilized ResNet50 (ImageNet weights) without the top layer (include_top=False) and applied global average pooling.

Each image is transformed into a 2048-dimensional feature vector.

2. Feature Selection
Implemented a Binary Artificial Algae Algorithm (Binary AAA) to identify the most relevant subset of features.

The algorithm balances classification accuracy with feature dimensionality.

Final selected subset: 1028 features.

3. Classification and Evaluation
Classifier: Voting Ensemble of Random Forest and XGBoost

Evaluation Method: 5-Fold Stratified Cross-Validation

Metrics reported:

Accuracy: 95.50%

Precision: 0.9556

Recall: 0.9550

F1 Score: 0.9550

AUC-ROC: 0.9944

(https://github.com/user-attachments/assets/9a9a6126-2e12-4b68-96b8-287fbb16e310)




