# IART Projects Repository

This repository contains coursework for the Artificial Intelligence class, including practical Jupyter Notebooks and two main projects:

- Project1: Fleet optimization for the Google Hash Code Self-Driving Rides problem
- Project2: Student performance prediction using Machine Learning and Deep Learning

## Repository Structure

- Notebooks/
  - Practice notebooks on search, optimization, clustering, and classification
- Project1/
  - Benchmark and sensitivity analysis material
  - Main code in Project1/Project Code/
- Project2/
  - Streamlit web app for student performance prediction
  - Training scripts for ML and DL models

## Project READMEs

- [Project1 README](Project1/Project%20Code/README.md)
- [Project2 README](Project2/README.md)

## General Requirements

- Python 3.10+ (recommended)
- pip
- Optional: virtual environment

Example virtual environment setup at repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Project 1: Self-Driving Rides Optimization

Main execution folder:

```bash
cd "Project1/Project Code"
```

Run main mode:

```bash
python3 main.py
```

Run comparative benchmark:

```bash
python3 benchmark.py
```

Notes:

- Input files are in Project1/Project Code/input/
- Benchmark outputs a CSV with score and execution time per algorithm

## Project 2: TutoringMaster (Student Performance Prediction)

Main execution folder:

```bash
cd Project2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models (required before opening the app):

```bash
python train_model.py
python train_dl_model.py
```

Run web app:

```bash
streamlit run app.py
```

## Notebooks

The Notebooks folder includes support material for:

- Search problem formulation
- Informed search
- Adversarial search
- Optimization
- Introduction to clustering
- Introduction to classification

## Contributors

Contributors identified in the repository:

- Goncalo Santos (up202306340)
- Joao Ferreira (up202305204)
- Manuel Pedro (up202303997)
