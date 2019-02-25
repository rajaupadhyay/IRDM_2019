import pickle
import numpy as np
import redis
from scipy.sparse import coo_matrix
import time
import sys
import random

# ['vocab_set', 'doc_name_index', 'vocab_dictionary']
conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())




def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


vocab_pickled = open('vocab_index.pickle', 'rb')
vocab_dict_pkl = pickle.load(vocab_pickled)
vocab_pickled.close()


print('Loading the document term matrix')
dtm_mat = open('doc_term_matrix.pickle', 'rb')
test_dtm = pickle.load(dtm_mat)
dtm_mat.close()
print('DOC-TERM Matrix loaded')

# Transposing DOC-TERM Matrix
print('Transposing DOC-TERM Matrix')
test_dtm_transposed = test_dtm.transpose()
test_dtm = None
# Convert to CSR
print('Converting to CSR Format')
test_dtm_transposed_csr = test_dtm_transposed.tocsr()
test_dtm_transposed = None
# Iterate over the vocabulary (building the index for each word)
cnt = 0
print('Beginning indexing')
prev = None


MAX_LENGTH_SO_FAR = -1
TOTAL_NON_ZERO_VALS_FOR_SPARSE_MTX = 0


# n_nonzero = 211785807 + 3922275
# # Dimensions of our Document-Term matrix will be len(docnames) X len(vocab)
# # Create single data, row and cols array - this will contain all the required data
# print('creating empty data, row and cols arrays')
# data = np.empty(n_nonzero, dtype=np.intc)
# rows = np.empty(n_nonzero, dtype=np.intc)
# cols = np.empty(n_nonzero, dtype=np.intc)
# print('created arrays')

ind = 0

for vocab_word in vocab_dict_pkl:
    # if cnt < 4:
    #     print(vocab_word)

    # if cnt % 100000 == 0:
    #     print(cnt, vocab_word)

    cnt += 1
    vocab_word_idx = conn.hget('vocab_dictionary', vocab_word)

    idx_for_word = int(vocab_word_idx)

    transposed_row_from_matrix = test_dtm_transposed_csr[idx_for_word]
    rows, cols = transposed_row_from_matrix.nonzero()

    # Cols contains a list of the document indices where the term (vocab_word) occurs
    cols = list(map(int, cols))
    # dta = transposed_row_from_matrix.data
    # dta = list(map(int, dta))

    # dlen_val is the total number of documents that this term occurs in
    dlen_val = len(cols)

    # cols.insert(0, dlen_val)
    # ########################
    # # +1 because you're storing the length in the first cell (length indicates the number of documents that this word appeared in)
    # ind_end = ind + dlen_val + 1
    #
    # data[ind:ind_end] = np.array(cols)
    # cols[ind:ind_end] = np.array([i_tr for i_tr in range(dlen_val+1)])
    # rows[ind:ind_end] = np.repeat(idx_for_word, dlen_val+1)
    #
    # ind = ind_end
    ########################

    # TOTAL_NON_ZERO_VALS_FOR_SPARSE_MTX += dlen_val

    # if dlen_val>MAX_LENGTH_SO_FAR:
    #     print('word appears in many docs', dlen_val, vocab_word)
    #     MAX_LENGTH_SO_FAR = dlen_val

    if dlen_val > 100000:
        print('word {} appears in {} docs - current idx {}'.format(vocab_word, dlen_val, cnt))


# print('building sparse inverted_index')
#
# iv_idx = coo_matrix((data, (rows, cols)), shape=(3922275, 1287377), dtype=np.intc)
#
# print('successfully built inverted_index!')
#
# print('pickling inverted_index')
# pickle_object(iv_idx, 'inverted_index')
# print('inverted_index READY!')















# print('Loading')
# tf_c_pkl = open('tf_idf_matrix_compressed.pickle', 'rb')
# tf_c = pickle.load(tf_c_pkl)
# tf_c_pkl.close()
# print('Loaded')
#
# print('Converting')
# tf_c = tf_c.tocsr()
# print('Converted')
#
# print('Retrieving')
# start = time.time()
# # rows = random.sample(range(2000000), 1000000)
# cols = random.sample(range(100000), 50)
#
# for i in range(200000):
#     if i%10000 == 0:
#         print(i)
#     res = tf_c[i, cols]
#
# end = time.time()
# total_time = end-start
# print(total_time)
# print('Retrieved')
