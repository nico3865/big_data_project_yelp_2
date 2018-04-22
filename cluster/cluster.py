import os
import sys
from numpy import array
from math import sqrt
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from  pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import desc

def main():
  spark = SparkSession.Builder().getOrCreate()
  # load dataset
  # datapath = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
  # dataset = spark.read.format('libsvm').json(datapath+'/data/business.json')

  filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/business_MTL_ONLY.json'
  # filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review_MTL_ONLY.json'
  dataset = spark.read.format('libsvm').json(filename)
  print(dataset)

  # get longitude and latitude
  ll = dataset.select(dataset.categories[0], dataset.longitude, dataset.latitude)
  ll = ll.withColumnRenamed('categories[0]', 'categories')

  ll.show()

  print(ll.schema.names)
  # for item in ll.schema.names:
  #   print(item)
  #   for item2 in item:
  #     print(item2)
  sys.exit()
  # convert ll to dense vectors
  # data =ll.rdd.map(lambda x:(Vectors.dense(float(x[0]), float(x[1])),)).collect()
  assembler = VectorAssembler(
      inputCols=['longitude', 'latitude'],
      outputCol='features')

  df = assembler.transform(ll)

  # set KMeans k and seed
  kmeans = KMeans(k=4, seed=1)

  # generate model
  model = kmeans.fit(df)

  # Make predictions
  predictions = model.transform(df)
  predictions.show(20)
  # Evaluate clustering by computing Silhouette score
  evaluator = ClusteringEvaluator()

  silhouette = evaluator.evaluate(predictions)
  print("Silhouette with squared euclidean distance = " + str(silhouette))

  # number of location in each cluster
  print('Number of business in each cluster: ')
  predictions.groupBy('prediction').count().sort(desc('count')).show()


  # show in which cluster do we have more restaurants
  print('Number of restaurant per clusters')
  predictions.where(predictions.categories == 'Restaurants').groupBy('prediction').count().sort(desc('count')).show()


  # Shows the result.
  centers = model.clusterCenters()
  print("Cluster Centers: ")
  for center in centers:
      print(center)

if __name__ == '__main__':
  main()