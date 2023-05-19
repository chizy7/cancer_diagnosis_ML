# Cancer Diagnosis using Machine Learning

This project applies three machine learning algorithms (Random Forest, Naive Bayes, and Decision Trees) to diagnose cancer based on clinical variables. The dataset contains 20 clinical variables and 569 samples, labeled as benign (B) or malignant (M).

The project is implemented using Python and PySpark MLlib.

## Prerequisites

- Python 3.8 or higher
- Apache Spark 3.0.0 or higher
- Pyspark Python library

## Installation

To set up your local environment:

1. Clone this repository to your local machine.
    ```bash
    git clone https://github.com/your_github_username/big_data_project.git
    ```
2. Go to the project directory.
    ```bash
    cd big_data_project
    ```
3. Install the necessary Python libraries specified in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. To run the project, go to the `src/` directory and run `main.py`:
    ```bash
    cd src/
    python main.py
    ```
2. The results (validation error, precision, recall, and F1 score) will be printed out for each model (Random Forest, Decision Tree, Naive Bayes).

## Notebooks

- The `notebooks/` directory contains Jupyter notebooks for exploratory data analysis (`exploration.ipynb`) and model training (`model_training.ipynb`). These notebooks provide a more detailed walkthrough of the project.

## Data

- The `data/` directory should contain the dataset. The dataset is expected to be in LibSVM format.
