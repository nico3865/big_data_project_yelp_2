import sys
import os
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer

from constants import SEED

def get_user_business(rating, user_mean, item_mean, rating_global_mean):
  return rating-(user_mean +item_mean-rating_global_mean)

def get_final_ratings(i, user_mean, item_mean, global_average_rating):
  final_ratings = i+user_mean+item_mean-global_average_rating
  return final_ratings

def main():
  spark = SparkSession.Builder().getOrCreate()
  seed = int(sys.argv[SEED])
  datapath = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
  rdd = spark.read.json(datapath+'/data/review.json').limit(100000).rdd
  df = spark.createDataFrame(rdd)
  (training, test) = df.randomSplit([0.8, 0.2], seed)
  userIdRdd1 = test.select('user_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))
  businessIdRdd1 = test.select('business_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))

  # convert to dataframe
  userIdDf2 = spark.createDataFrame(userIdRdd1)\
                  .withColumnRenamed('_1', 'user_id') \
                  .withColumnRenamed('_2', 'user_id_indexed')
  businessIdDf2 = spark.createDataFrame(businessIdRdd1) \
                      .withColumnRenamed('_1', 'business_id') \
                      .withColumnRenamed('_2', 'business_id_indexed')

  # join user id zipped with index and business id with index
  test = test.join(userIdDf2, ['user_id'], 'left').join(businessIdDf2, ['business_id'], 'left')

  # get user mean
  user_mean = training.groupBy('user_id').mean('stars').withColumnRenamed('avg(stars)', 'user-mean')

  # get item mean
  business_mean = training.groupBy('business_id').mean('stars').withColumnRenamed('avg(stars)', 'business-mean')

  # join user mean df and training df
  training = training.join(user_mean, ['user_id']) \
          .select(training['user_id'], training['business_id'], training['stars'], user_mean['user-mean'])

  # join item mean df and traning df
  training = training.join(business_mean, ['business_id']) \
          .select(training['user_id'], training['business_id'], training['stars'],
                  user_mean['user-mean'], business_mean['business-mean'])

  # get global average
  rating_global_average = training.groupBy().avg('stars').head()[0]

  # add user item interaction to training column
  training = training.withColumn('user-business-interaction',
                                  get_user_business(training['stars'],
                                                user_mean['user-mean'],
                                                business_mean['business-mean'],
                                                rating_global_average))

  # convert distinct user ids and business ids to integer
  userIdRdd = training.select('user_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))
  businessIdRdd = training.select('business_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))

  # convert to dataframe
  userIdDf = spark.createDataFrame(userIdRdd)\
                  .withColumnRenamed('_1', 'user_id') \
                  .withColumnRenamed('_2', 'user_id_indexed')
  businessIdDf = spark.createDataFrame(businessIdRdd) \
                      .withColumnRenamed('_1', 'business_id') \
                      .withColumnRenamed('_2', 'business_id_indexed')
  # join user id zipped with index and business id with index
  training = training.join(userIdDf, ['user_id'], 'left').join(businessIdDf, ['business_id'], 'left')
  als = ALS(maxIter=5,
            rank=70,
            regParam=0.01,
            userCol='user_id_indexed',
            itemCol='business_id_indexed',
            ratingCol='user-business-interaction',
            coldStartStrategy='drop')
  als.setSeed(seed)
  model = als.fit(training)

  # Evaluate the model by computing the RMSE on the test data
  predictions = model.transform(test)
  
  predictions = predictions.join(user_mean, ['user_id'],'left')
  predictions = predictions.join(business_mean, ['business_id'], 'left')
  rating_global_mean = training.groupBy().mean('stars').head()[0]
  predictions = predictions.na.fill(rating_global_mean)
  final_stars = predictions.withColumn('final-stars', get_final_ratings(predictions['prediction'],
                                          predictions['user-mean'],
                                          predictions['business-mean'],
                                          rating_global_mean))
  high_stars = final_stars.where(final_stars['final-stars'] >= 3)
  low_stars = final_stars.where(final_stars['final-stars'] < 3)
  
  evaluator = RegressionEvaluator(metricName='rmse',
                                  labelCol='stars',
                                  predictionCol='final-stars')

  final_stars_rmse = evaluator.evaluate(final_stars)
  print('final stars rmse', float(final_stars_rmse))
  
  high_stars_rmse = evaluator.evaluate(high_stars)
  print('number of high stars', high_stars.count())
  print('high stars rmse', float(high_stars_rmse))

  print('number of low stars', low_stars.count())
  low_stars_rmse = evaluator.evaluate(low_stars)
  print('low stars rmse', float(low_stars_rmse))

if __name__ == '__main__':
    main()