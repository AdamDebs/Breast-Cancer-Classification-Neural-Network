Biomedical Triage System
This project focuses on the intersection of medical hardware reliability and AI-driven diagnostics. I developed a clinical decision-support tool to classify breast biopsy samples as benign or malignant using Python and PyTorch.

Project Overview
The goal of this system is to assist in clinical triage by identifying high-risk samples. In a medical setting, a False Negative (missed cancer) is significantly more costly than a False Positive. This project demonstrates how to optimise a model to prioritise patient safety and clinical accuracy.

Key Technical Highlights:
Clinical Machine Learning (PyTorch): Architected a 16-8-1 Feedforward Neural Network to classify samples based on nuclear characteristics.

Safety Optimisation: Implemented a 0.3 clinical threshold to prioritise Malignant Recall, resulting in a 38% improvement in patient safety by reducing missed malignancy cases.

Logic & Troubleshooting: The system uses the same troubleshooting mindset I apply to my business, Debs Solutions & Co., where I developed AI voice agents with emergency-detection protocols.

How to Use
Open the Biomedical_Triage_System.ipynb file in Google Colab or Jupyter Notebook.

Ensure you have the required clinical dataset uploaded to your environment.

Run the cells from top to bottom to see the exploratory data analysis, the logistic regression baseline, and the final neural network evaluation.

Tools Used
Python (Pandas, NumPy, Scikit-learn)

PyTorch (Neural Network Architecture)

Matplotlib & Seaborn (Data Visualisation)
