# Cancer Diagnosis using Machine Learning with PySpark

This repository contains code for a project on diagnosing cancer using machine learning techniques in PySpark. The project uses the Random Forest, Naive Bayes, and Decision Tree algorithms to train and evaluate prediction models, and compares their performances. The data used consists of 569 samples, each with 20 clinical features and a label indicating whether the sample is benign (B) or malignant (M).

## Getting Started

To run the project, you will need to have Python and PySpark installed.

### Directory Structure

The project's directory structure is as follows:

* `main.py`: The main script to run the project.
* `src/`: Contains separate Python scripts for each algorithm (`random_forest.py`, `naive_bayes.py`, and `decision_tree.py`).
* `data/`: Contains the cancer dataset in CSV format (`cancer_dataset.csv`).

## Running the Project

The project can be run using the `main.py` script. This script does the following:

1. Initializes a SparkContext.
2. Loads the data as a DataFrame.
3. Transforms the data to the format of LabeledPoint(label, features).
4. Splits the data into training, validation, and testing sets.
5. Runs each of the algorithms on the data.
6. Stops the SparkContext.

To run the project, navigate to the project's root directory and run the following command:
```
python3 main.py
```

## Algorithms

The project uses the following three machine learning algorithms to classify the samples:

1. **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees at training time and outputting the class that is the mode of the classes (classification) of the individual trees.

2. **Naive Bayes**: A classification technique based on Bayes' theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

3. **Decision Tree**: A decision support tool that uses a tree-like model of decisions and their possible consequences. It is one way to display an algorithm that only contains conditional control statements.

Each algorithm's script contains a function run(sc, train_data, val_data, test_data) that trains a model on the training data, makes predictions on the validation data, and saves the trained model.

<!-- ## License

This project is licensed under the MIT License - see the LICENSE.md file for details. -->

## Acknowledgments

The project uses the PySpark MLLib library. See [Spark MLLib](https://spark.apache.org/mllib/) for more details.