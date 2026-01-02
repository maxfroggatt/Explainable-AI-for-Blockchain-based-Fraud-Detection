# Explainable-AI-for-Blockchain-based-Fraud-Detection
A Python-based Explainable AI (XAI) project for blockchain fraud detection. Includes data preprocessing, feature engineering, model training, SHAP/LIME explainability, and optional smart contract logging via Ganache.

1: Download Data
Download the Elliptic2 dataset https://www.kaggle.com/datasets/ellipticco/elliptic2-data-set. Please download, unzip, and put them in a folder "dataset". The downloaded files should look like:

dataset

├── background_edges.csv

├── background_nodes.csv

├── connected_components.csv

├── edges.csv

└── nodes.csv

2: pip install -r requirements.txt

3: python src/train_model.py

4: python src/explain_shap.py