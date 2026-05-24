# 🎓 TutoringMaster - Student Performance Prediction System

## 📖 Project Summary
TutoringMaster is a machine learning-powered web application designed to predict a student's performance (Grade Class) and assess their academic risk level. It takes into account various student factors such as weekly study time, absences, parental support, extracurricular activities, and demographic information.

The application features a side-by-side comparative analysis between two different AI approaches:
- **Classical Machine Learning**
- **Deep Learning Neural Network**

Based on the predictions, the system also provides actionable recommendations to help evaluate and improve the student's academic performance.

## 🚀 Getting Started

To run the web version of this project locally, you need to train the AI models first before launching the Streamlit app.

### 1. Train the AI Models
Before starting the web application, you **must** run both training scripts. These scripts will process your dataset (`dados_grade_invertida.csv`) and generate the required model files (`model.pkl`, `dl_model.pth`, and `dl_preprocessor.pkl`) that the web app relies on.

Run the following commands in your terminal:

```bash
# Train the Classical Machine Learning model
python train_model.py

# Train the Deep Learning Neural Network model
python train_dl_model.py
```

### 2. Run the Web Application
Once the training scripts have successfully finished and generated the model files, you can start the Streamlit web interface.

Run the following command:

```bash
streamlit run app.py
```

This will automatically open the application in your default web browser (usually at `http://localhost:8501`).

## 📦 Requirements
Make sure you have the required Python libraries installed to run both the training scripts and the frontend app. Based on the project, you will need:
- `streamlit`
- `pandas`
- `scikit-learn`
- `torch` (PyTorch)
- `joblib`
- `matplotlib`

## 👥 Contributors
- Gonçalo Santos (up202306340)
- João Ferreira (up202305204)
- Manuel Pedro (up202303997)
