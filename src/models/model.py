from pyspark import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

def train_and_evaluate_model():
    sc = SparkContext(appName="CancerDiagnosis")
    data = MLUtils.loadLibSVMFile(sc, "/data/processed/cancer_data_preprocessed.csv")
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model
    rf_model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                         numTrees=3, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=4, maxBins=32)

    # Train a NaiveBayes model
    nb_model = NaiveBayes.train(trainingData, 1.0)

    # Train a DecisionTree model
    dt_model = DecisionTree.trainClassifier(trainingData,numClasses=2, categoricalFeaturesInfo={},
                                            impurity='gini', maxDepth=5, maxBins=32)

    # Evaluate models on test instances and compute test error
    predictions_rf = rf_model.predict(testData.map(lambda x: x.features))
    labels_and_predictions_rf = testData.map(lambda lp: lp.label).zip(predictions_rf)
    metrics_rf = MulticlassMetrics(labels_and_predictions_rf)

    predictions_nb = nb_model.predict(testData.map(lambda x: x.features))
    labels_and_predictions_nb = testData.map(lambda lp: lp.label).zip(predictions_nb)
    metrics_nb = MulticlassMetrics(labels_and_predictions_nb)

    # Evaluate model on test instances and compute test error
    predictions_dt = dt_model.predict(testData.map(lambda x: x.features))
    labels_and_predictions_dt = testData.map(lambda lp: lp.label).zip(predictions_dt)
    metrics_dt = MulticlassMetrics(labels_and_predictions_dt)


    # Calculate metrics
    def calculate_metrics(metrics):
        precision = metrics.precision()
        recall = metrics.recall()
        f1Score = metrics.fMeasure()
        return precision, recall, f1Score

    precision_rf, recall_rf, f1Score_rf = calculate_metrics(metrics_rf)
    precision_nb, recall_nb, f1Score_nb = calculate_metrics(metrics_nb)
    precision_dt, recall_dt, f1Score_dt = calculate_metrics(metrics_dt)

    # Save models
    rf_model.save(sc, "target/tmp/myRandomForestClassificationModel")
    nb_model.save(sc, "target/tmp/myNaiveBayesModel")
    dt_model.save(sc, "target/tmp/myDecisionTreeClassificationModel")

    # Stop the SparkContext
    sc.stop()

    # Print results
    print("Random Forest - Precision: {}, Recall: {}, F1 Score: {}".format(precision_rf, recall_rf, f1Score_rf))
    print("Naive Bayes - Precision: {}, Recall: {}, F1 Score: {}".format(precision_nb, recall_nb, f1Score_nb))
    print("Decision Tree - Precision: {}, Recall: {}, F1 Score: {}".format(precision_dt, recall_dt, f1Score_dt))

if __name__ == "__main__":
    train_and_evaluate_model()
