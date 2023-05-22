from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.evaluation import MulticlassMetrics

def run(sc, train_data, val_data, test_data):
    # Train the model
    model = NaiveBayes.train(train_data, 1.0)

    # Make predictions on the validation data
    predictions = model.predict(val_data.map(lambda x: x.features))

    # Evaluation
    labels_and_preds = val_data.map(lambda p: p.label).zip(predictions)
    accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(val_data.count())
    print('Naive Bayes Validation Accuracy =', accuracy)

    # Save the model
    model.save(sc, "src/naive_bayes_model")
