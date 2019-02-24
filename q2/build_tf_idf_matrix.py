import pickle
import numpy as np
import redis
from scipy.sparse import coo_matrix

# conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
# print(conn.keys())
# print(conn.hget('vocab_dictionary', '1751'))
# print(conn.hget('doc_name_index', 'Thomas_Hinton_-LRB-priest-RRB-'))

def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


# print('Loading the doc term matrix.')
#
# dtm_mat = open('doc_term_matrix.pickle', 'rb')
# test_dtm = pickle.load(dtm_mat)
# dtm_mat.close()
#
# # print(test_dtm.getrow(5090422))
#
#
# # row_sums = test_dtm.sum(1)
# # idx = np.where(row_sums == 0)[0]
# #
# # print(idx[:5])
#
# # print(test_dtm.getrow(178))
# print(test_dtm.shape)
#
#
# print('Loaded the doc term matrix successfully.')
#
# '''
# Calculate the inverse document frequency
# '''
# print('Calculating the document frequency')
# document_frequency = test_dtm.col
# uniq_indices, document_frequency = np.unique(document_frequency, return_counts=True)
# # document_frequency = np.count_nonzero(test_dtm.toarray(), axis=0)
#
# print('Calculating the inverse document frequency')
# # log_10(N/n)   -> np.log10(5/document_frequency)
# idf_vector = np.log10(test_dtm.shape[0]/document_frequency)
#
# print('Pickling the IDF vector')
# pickle_object(idf_vector, 'idf_vector')
#
# document_frequency = None
#
# # print(idf_vector)
#
# '''
# Now normalise the term frequencies
# '''
# # print(test_dtm.sum(1))
# print('Normalising term frequencies')
#
# print('Retrieving nonzero vals')
# r,c = test_dtm.nonzero()
# row_sums = test_dtm.sum(1)
#
# print('count non zeroes', np.count_nonzero(row_sums))
#
#
# print('Normalising log')
# inverse = ((1.0/row_sums)[r])
#
# inverse = np.squeeze(np.asarray(inverse))
# # print(inverse)
#
# print('Building normalised matrix')
# div_sp = coo_matrix((inverse, (r,c)), shape=(test_dtm.shape), dtype=np.float16)
# out = test_dtm.multiply(div_sp)
#
# print('Pickling the normalised term frequencies')
# pickle_object(out, 'normalised_term_freqs')
#
#
# # print(out.toarray())
# '''
# prepare tf-idf matrix
# '''
# print('Preparing the TF-IDF matrix')
# tf_idf_vectors = out.multiply(idf_vector)
#
# print('Pickling the TF-IDF matrix!')
# pickle_object(tf_idf_vectors, 'tf_idf_matrix_compressed')





'''
Test that the tf_idf_vector has been calculated correctly
'''

def test_tf_idf(file_id):
    # start with getting the id of a sample document from one of the docs e.g. 1928_in_association_football
    # Get the index of that document in the matrix (for the above example: 3917198)

    conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
    val = int(conn.hget('doc_name_index', file_id))

    # Lets retrieve the tf-idf vector for this document first from the tf-idf matrix
    final_tf_idf_res = None
    with open('tf_idf_matrix_compressed.pickle', 'rb') as input_file:
        print('loading tf-idf')
        tfidf = pickle.load(input_file)
        final_tf_idf_res = tfidf.getrow(val)
        print(final_tf_idf_res)
    # Lets get the vector for this document from the doc_term_matrix
    with open('doc_term_matrix.pickle', 'rb') as input_file:
        doc_term_matrix = pickle.load(input_file)
        req_row = doc_term_matrix.getrow(val)

        col_indices_for_vocab = np.split(req_row.indices, req_row.indptr)[1:-1]


        document_frequency = doc_term_matrix.col
        uniq_indices, document_frequency = np.unique(document_frequency, return_counts=True)

        # Get respective idf values for the terms in this doc
        idf_vec = []
        for tkn_i in col_indices_for_vocab[0]:
            idf_val = document_frequency[tkn_i]
            idf_vec.append(idf_val)

        idf_vec = np.array(idf_vec)
        idf_vec = np.log10(5396041/idf_vec)

        # Get the normalised term frequency
        data_items = req_row.data
        data_items = data_items/(data_items.sum())

        tf_idf_vec = np.multiply(data_items, idf_vec)
        print(tf_idf_vec)


# test_tf_idf('Thomas_Hinton_-LRB-priest-RRB-')


'''
4780465

(0, 3513252)	0.1338629612254747   september
  (0, 3311484)	0.30301128023099055   windsor
  (0, 3052358)	0.18921602682634772    Thomas
  (0, 2805768)	0.6775284939260445      1757
  (0, 1455401)	0.1427522719954461    3
  (0, 1144118)	0.14538223330145253   died
  (0, 912197)	0.38324141827839353   hinton
  (0, 799649)	0.34315121107816754   1751
  (0, 430288)	0.2743381744432157   canon

  Thomas Hinton -LRB- died 3 September 1757 -RRB- was a Canon of Windsor from 1751 to 1757

[0.13389565 0.30308528 0.18926223 0.67769395 0.14278713 0.14541774
 0.38333501 0.34323501 0.27440517]
'''



# Convert the dtype of matrix elements: to np.float16
# 1694286456 (original size)
# 423571614 (compressed)

tfidfm_f = open('tf_idf_matrix_compressed.pickle', 'rb')
tfidf_m = pickle.load(tfidfm_f)
# orig = tfidf_m.data.nbytes
tfidfm_f.close()
tfidf_m = tfidf_m.astype(np.float16)
# new_size = tfidf_m.data.nbytes
print('pickling')
pickle_object(tfidf_m, 'tf_idf_matrix_comp')
