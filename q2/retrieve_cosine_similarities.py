import pickle
import numpy as np
import redis
from scipy.sparse import coo_matrix
import time

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())

dummy_query = 'fyssoun and elldridg'

print(conn.zcard('ividx_netsingl'))
# print(conn.zrange('ividx_elldridg', 0, -1, withscores=True))



# GIVEN A QUERY RETRIEVE THE RELEVANT VECTORS FROM THE INDEX

Original sparse matrix with TFIDF values
>>> mtr.toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])

# Convert to csr format
# Ensure dtype is np.intc
>>> mtr_csr.toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]], dtype=int64)


# For each of the documents in the inverted index, get the words from the respective indices (depending on the words of the query)
# Only get the words that are in the query
# Step1: Get the index of the words in the query from the vocab index
# first part of this call is the document and the next part are the indices
# you can supply all the document indices at once as well
# Then you have a matrix with rows as the relevant documents and the columns as the relevant words in those documents relating to the query
mtr_csr[0, [0,2,3]]

# Carry out a dot product with the tfidf vector for the query
>>> s1.toarray()
array([[4, 9, 0]], dtype=int64)
>>> s1 = _
>>> np.squeeze(s1)
array([4, 9, 0], dtype=int64)

# Perform cosine similarity for each doc vector with the query vector of tfidf values
