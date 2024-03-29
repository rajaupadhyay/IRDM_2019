import pickle
import numpy as np
import redis
from scipy.sparse import coo_matrix
import time
import sys

# ['vocab_set', 'doc_name_index', 'vocab_dictionary']
conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
# print(conn.keys())
# print(conn.hget('vocab_dictionary', 'hp18c'))

# 'ividx_isenau', 'ividx_gtwo', 'ividx_kūshgūnlū', 'ividx_cephalotorax', 'ividx_nacllik', 'ividx_xhlyfm', 'ividx_68386', 'ividx_streinu'
print(conn.dbsize())
# print(conn.zcard('ividx_isenau'))
# print(conn.zrange('ividx_isenau', 0, -1, withscores=True))


def delete_keys():
    print('DELETING KEYS')
    # time.sleep(10)
    for k_val in conn.scan_iter(match='ividx_*'):
        conn.delete(k_val)

# delete_keys()

def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)




vocab_pickled = open('vocab_index.pickle', 'rb')
vocab_dict_pkl = pickle.load(vocab_pickled)
vocab_pickled.close()

# def pickleLoader(pklFile):
#     try:
#         while True:
#             yield pickle.load(pklFile)
#     except EOFError:
#         print('error in loading pickle')
#         pass

'''
310400
Error 32 while writing to socket. Broken pipe.
19742009
previous κλυδωνα
'''

print('Loading the document term matrix')
dtm_mat = open('doc_term_matrix.pickle', 'rb')
test_dtm = pickle.load(dtm_mat)
dtm_mat.close()
print('DOC-TERM Matrix loaded')

# Transposing DOC-TERM Matrix
print('Transposing DOC-TERM Matrix')
test_dtm_transposed = test_dtm.transpose()

# Convert to CSR
print('Converting to CSR Format')
test_dtm_transposed_csr = test_dtm_transposed.tocsr()

# Iterate over the vocabulary (building the index for each word)
cnt = 0
print('Beginning indexing')
prev = None

pipe = conn.pipeline()

starting_point=0
new_start_point = 310299

def execute_pipeline(pipe):
    pipe.execute()
    pipe = conn.pipeline()
    return pipe


for vocab_word in vocab_dict_pkl:
    # if cnt < 4:
    #     print(vocab_word)
    cnt += 1

    # if vocab_word != 'κλυδωνα' and starting_point == 0:
    #     continue
    # else:
    #     starting_point = 1

    if cnt < new_start_point:
        continue

    if cnt == new_start_point+100 or cnt == new_start_point+101:
        print(vocab_word)

    vocab_word_idx = conn.hget('vocab_dictionary', vocab_word)

    # cnt += 1
    if cnt%10 == 0:
        try:
            pipe = execute_pipeline(pipe)
            # pipe.execute()
            # pipe = conn.pipeline()
        except Exception as e:
            print(cnt)
            print('size of pipe', sys.getsizeof(pipe))
            print(e)
            print(vocab_word)
            print('previous', prev)
            break

    if cnt%500 == 0:
        time.sleep(2)

    if cnt%2500 == 0:
        time.sleep(10)

    if cnt % 100000 == 0:
        print('index', cnt)

    set_key_name = 'ividx_{}'.format(vocab_word)

    idx_for_word = int(vocab_word_idx)

    transposed_row_from_matrix = test_dtm_transposed_csr[idx_for_word]
    rows, cols = transposed_row_from_matrix.nonzero()
    cols = list(map(str, cols))
    dta = transposed_row_from_matrix.data
    dta = list(map(int, dta))

    inner_values_for_set = dict(list(zip(cols, dta)))
    dlen_val = len(inner_values_for_set)

    if dlen_val > 300000:
        pipe = execute_pipeline(pipe)

    pipe.zadd(set_key_name, inner_values_for_set)

    # try:
    #     pipe.zadd(set_key_name, inner_values_for_set)
    # except Exception as e:
    #     print(cnt)
    #     print(e)
    #     print(vocab_word)
    #     print('previous', prev)
    #     break
    prev = vocab_word
