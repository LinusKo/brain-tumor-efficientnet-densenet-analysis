# brain-tumor-efficientnet-densenet-analysis

Project Overview.
This repository presents a comprehensive study on the use of deep learning for the classification of brain tumors from MRI images. The core of this project is a comparative analysis of two state-of-the-art convolutional neural network architectures—EfficientNetB0 and DenseNet121—to determine the most effective model for this critical medical task. The goal is to provide a reliable tool for automated primary screening, aiding specialists and potentially improving diagnostic efficiency.

Methodology.
- Dataset: We used the publicly available Brain Tumor MRI Dataset from Kaggle, which contains four classes: pituitary, glioma, meningioma, and no_tumor
  ( https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
- Approach: Transfer learning was applied to both models, leveraging weights pre-trained on the large-scale ImageNet dataset to accelerate training and improve performance on our specific task.
- Data Handling: To prevent overfitting and enhance the models' generalization capabilities, we employed extensive data augmentation techniques. All images were pre-processed and resized to a consistent dimension for model compatibility.
- Training: Both models were trained under identical conditions, using advanced callbacks such as EarlyStopping and ReduceLROnPlateau to ensure an optimal and fair training process.

Performance Comparison
The models were rigorously evaluated on a held-out test set, with the following key performance metrics recorded:

Metric    EfficientNetB0  DenseNet121
Accuracy     99.31%         98.45%
Precision     99%            98%
Recall        99%            98%
F1-Score	    99%            98%

- Conclusion: The EfficientNetB0 model demonstrated superior performance across all key metrics, making it the more suitable choice for this specific classification task.

Repository Structure
- notebooks/ - Contains the Jupyter Notebooks used for training and evaluating both models.
- README.md - This file, providing a comprehensive project overview.

How to Reproduce the Results
The trained models are not included in this repository due to their large file size. However, you can easily reproduce the results by following these steps:
1. Download the Dataset: The dataset is publicly available on Kaggle at this link (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
2. Run the Notebooks:
- The efficientnet_training.ipynb notebook was developed for and can be run on Google Colaboratory.
- The densenet_training.ipynb notebook was developed for and can be run on Kaggle Notebooks.

Trained Models
For convenience, the final trained models are also available for download on Hugging Face Hub.
- EfficientNetB0: https://huggingface.co/linakosmina/efficientnetb0-brain-tumor-classification
- DenseNet121: https://huggingface.co/linakosmina/densenet121-brain-tumor-classification

Technologies Used
- Python
- TensorFlow / Keras
- Numpy
- Matplotlib
- Seaborn

License
This project is licensed under the Apache License 2.0.
