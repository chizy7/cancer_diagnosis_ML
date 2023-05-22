from pyspark import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import when
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from src import random_forest, naive_bayes, decision_tree

# Initialize SparkContext
sc = SparkContext(appName="CancerDiagnosis")

# Create SQLContext from SparkContext
sqlContext = SQLContext(sc)

# Load the data
data = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('data/cancer_dataset.csv')

# Convert the string labels to float
data = data.withColumn("diagnosis", when(data["diagnosis"] == 'M', 1.0).otherwise(0.0))

# Convert the DataFrame back to RDD
data_rdd = data.rdd.map(lambda x: LabeledPoint(x[-1], Vectors.dense(x[:-1])))

# Split the data into training, validation, and test sets
train_data, val_data, test_data = data_rdd.randomSplit([0.6, 0.2, 0.2])

# Execute algorithms
random_forest.run(sc, train_data, val_data, test_data)
naive_bayes.run(sc, train_data, val_data, test_data)
decision_tree.run(sc, train_data, val_data, test_data)

# Stop SparkContext
sc.stop()
