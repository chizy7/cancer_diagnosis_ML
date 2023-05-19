from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import RandomForest, DecisionTreeModel
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics

# Load data function
def load_data(path):
    data = MLUtils.loadLibSVMFile(sc, path)
    # Split the data into training, validation, and test sets (60%, 20%, 20%)
    splits = data.randomSplit([0.6, 0.2, 0.2])
    return splits[0], splits[1], splits[2]

# Training and evaluation function
def train_and_evaluate_model(algorithm, trainingData, validationData, testData):
    if algorithm == "RandomForest":
        model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                             numTrees=10, featureSubsetStrategy="auto",
                                             impurity='gini', maxDepth=4, maxBins=32)
    elif algorithm == "NaiveBayes":
        model = NaiveBayes.train(trainingData)
    elif algorithm == "DecisionTree":
        model = DecisionTreeModel.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                                  impurity='gini', maxDepth=4, maxBins=32)
    # Evaluate model on validation set
    predictionsAndLabels = validationData.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    validationError = 1.0 * predictionsAndLabels.filter(lambda pl: pl[0] != pl[1]).count() / validationData.count()

    # Evaluate model on test set
    predictionsAndLabels_test = testData.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    metrics = MulticlassMetrics(predictionsAndLabels_test)

    # Compute precision, recall, f1 score
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()

    return model, validationError, precision, recall, f1Score

if __name__ == "__main__":
    conf = SparkConf().setAppName("Cancer Diagnosis ML")
    sc = SparkContext(conf=conf)

    # trainingData, validationData, testData = load_data("data/cancer_data.csv")
    trainingData, validationData, testData = load_data("../data/cancer_data.csv")


    algorithms = ["RandomForest", "DecisionTree", "NaiveBayes"]
    for algorithm in algorithms:
        print(f"\nTraining, validating and testing {algorithm}")
        model, validationError, precision, recall, f1Score = train_and_evaluate_model(algorithm, trainingData, validationData, testData)
        print(f"Validation Error: {validationError}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
        print(f"Test F1 Score: {f1Score}")
