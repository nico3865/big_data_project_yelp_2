# -*- coding: utf-8 -*-


from pandas import json

from pyspark.sql import SparkSession

filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review_1M.json'
# filename = '../data/review_1M.json'
# filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review_MTL_ONLY.json'
# filename = '/Users/nicolasg-chausseau/big_data_project_yelp/data/review_truncated_RAW.json'



# # spark = SparkSession.Builder().getOrCreate()
# # rdd = spark.read.json(filename).limit(1000000).rdd # datapath+'/data/review_trunca®ted_RAW.json'
# json_data=open(filename).read()
# data = json.loads(json_data)
# # pprint(data)




# import numpy as np
# import itertools
# with open(filename) as f_in:
#     x = np.genfromtxt(itertools.islice(f_in, 1, 12, None), dtype='unicode')
#     print(x[0,:])



# lines = []
# counter = 0
# filename_counter = 0
# with open(filename) as f:
#     for line in f:
#         counter += 1
#         lines += [line]
#         if counter > 50000 - 1:
#             outF = open("review_50K_"+str(filename_counter)+".json", "w")
#             outF.writelines(lines)
#             outF.close()
#             filename_counter += 1
#             counter = 0
#             lines = []





from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

spark = SparkSession.Builder().getOrCreate()

rows = spark.sparkContext.parallelize([
    Vectors.sparse(5, {1: 1.0, 3: 7.0}),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
])

mat = RowMatrix(rows)
# Compute the top 4 principal components.
# Principal components are stored in a local dense matrix.
pc = mat.computePrincipalComponents(4)
print(pc) # DenseMatrix([[-4.48591721e-01, -2.84238082e-01,  8.34454526e-02,
# pc.show()


# DenseMatrix([
#              [-4.48591721e-01, -2.84238082e-01,  8.34454526e-02, 8.36410201e-01],
#              [ 1.33019857e-01, -5.62115590e-02,  4.42397926e-02, 1.72243378e-01],
#              [-1.25231564e-01,  7.63626477e-01, -5.78071229e-01, 2.55415489e-01],
#              [ 2.16507567e-01, -5.65295877e-01, -7.95540506e-01, 4.85812143e-05],
#              [-8.47651293e-01, -1.15603405e-01, -1.55011789e-01, -4.53335549e-01]
#           ])

# Project the rows to the linear space spanned by the top 4 principal components.
projected = mat.multiply(pc)
projected.rows.map(lambda x: (x, )).toDF().show()