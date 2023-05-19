from pyspark import SparkContext
from pyspark.mllib.util import MLUtils

def load_and_preprocess_data():
    sc = SparkContext(appName="CancerDiagnosis")
    data = MLUtils.loadLibSVMFile(sc, "data/cancer_dataset.csv")

    # Perform necessary preprocessing like handling missing values, categorical variables, etc.
    # Then save the preprocessed data
    MLUtils.saveAsLibSVMFile(data, "data/processed/cancer_dataset_preprocessed.csv")

if __name__ == "__main__":
    load_and_preprocess_data()