# leaf_disease_classification_project

This repository contains a deep learning project focused on classifying various tomato leaf diseases using convolutional neural network architectures. The goal is to build accurate, efficient, and comparable models using several state-of-the-art backbones.

🌿 Project Overview

Plant diseases significantly impact agricultural productivity. Early detection can help prevent major crop losses. This project uses image-based classification to identify tomato leaf diseases.

We trained and evaluated three modern CNN architectures:

DenseNet121
ResNet50
EfficientNetB0

All models were trained on a dataset of tomato plant leaf images labeled with different disease categories.

# Requirements
pip install utils \n
pip install tensorflow keras \n
pip install numpy 

# Clone the repository
git clone https://github.com/duellieren/leaf_disease_classification_project.git
cd leaf_disease_classification_project

# Structure
leaf_disease_classification_project/
├── data/
│ └── raw/
│   ├── all_data/ 
│   ├── train/ 
│   └── test/ 
├── models/ 
├── notebooks/ 
├── utils/
└── README.md

# Working 
To use the project run the data_acquisition notebook to download the dataset and then run each of the model notebooks to perform the training and evaluation of the results.
