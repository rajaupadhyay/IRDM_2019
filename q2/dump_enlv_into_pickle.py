import pickle
from sqlitedict import SqliteDict
import zlib
import numpy as np
import time
import redis

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)


# def decompress_set(obj):
#     return pickle.loads(zlib.decompress(bytes(obj)))
#
#
#
# res = []
# with SqliteDict('enlv_for_docs.sqlite', decode=decompress_set) as enlv_dct:
#     total_keys = len(enlv_dct)
#     print(total_keys)
#
#     start_time = time.time()
#     for i in range(total_keys):
#         if i%100000 == 0:
#             end_time = time.time()
#             print(i, end_time-start_time)
#             start_time = time.time()
#         res.append(enlv_dct[i])
#
# res = np.array(res, dtype=np.float32)
#
# with open('enlv_for_documents.pickle', 'wb') as handle:
#     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_tf_idf(file_id):
    # start with getting the id of a sample document from one of the docs e.g. 1928_in_association_football
    # Get the index of that document in the matrix (for the above example: 3917198)

    conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
    val = int(conn.hget('doc_name_index', file_id))

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
        print(data_items)
        data_items = data_items/(data_items.sum())
        print(data_items)

        tf_idf_vec = np.multiply(data_items, idf_vec)
        print(tf_idf_vec)


test_tf_idf('Nikolaj')




# with open('inverted_doc_name_dict.pickle', 'rb') as handle:
#     han = pickle.load(handle)
#     print(han['3568862'])
