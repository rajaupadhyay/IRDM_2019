import pickle
import numpy as np
import redis
from scipy.sparse import coo_matrix

def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


# conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
# print(conn.keys())
# print(conn.hget('vocab_dictionary', 'won'))

dtm_mat = open('doc_term_matrix.pickle', 'rb')
test_dtm = pickle.load(dtm_mat)
dtm_mat.close()

print('Calculating word freq over collection')
# word frequency in collection
col_sums = test_dtm.sum(0)
print('Converting to np.array and squeezing')
col_sums = np.squeeze(np.array(col_sums, dtype='int32'))

print('Pickling object')
pickle_object(col_sums, 'wordFrequencyInCollection')


# # print(test_dtm.getrow(42748))
# # 1963_Estonian_SSR_Football_Championship  42748
#
# row_sums = test_dtm.sum(1)
# test_dtm = None
# print(row_sums.shape)
#
#
# row_sums = np.array(row_sums).reshape(-1,).tolist()
# print(len(row_sums))
#
# print(row_sums[42748])
# pickle_object(row_sums, 'doc_length_vector')

# print(type(row_sums))
# row_sums = row_sums.flatten()
# print(row_sums.shape)
#
# row_sums = np.squeeze(row_sums)
# print(row_sums.shape)
#
# print(row_sums[42748])


# dl_v_f = open('doc_length_vector.pickle', 'rb')
# dl_v = pickle.load(dl_v_f)
# dl_v_f.close()
#
# print(dl_v[42748])
