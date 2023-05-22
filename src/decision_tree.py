from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.evaluation import MulticlassMetrics

def run(sc, train_data, val_data, test_data):
    # Train the model
    model = DecisionTree.trainClassifier(train_data, numClasses=2, categoricalFeaturesInfo={},
                                         impurity='gini', maxDepth=5, maxBins=32)

    # Make predictions on the validation data
    predictions = model.predict(val_data.map(lambda x: x.features))

    # Evaluation
    labels_and_preds = val_data.map(lambda p: p.label).zip(predictions)
    accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(val_data.count())
    print('Decision Tree Validation Accuracy =', accuracy)

    # Save the model
    model.save(sc, "src/decision_tree_model")
