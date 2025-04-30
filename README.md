# Heart-Rate-Monitoring
Repository for implementation of my 3rd year project

data_processing.ipynb - preprocessing, takes the data from MIT-BIH applies preprcessing and produces train test and validation sets. Train set is augmented with SMOTE-Tomek.
RAT-Net.ipynb - Implentation of RAT-Net and code for training and testing models. Needs data splits from data_processing.ipynb. 

Both are made for use in google colab and require some refractoring to run locally
