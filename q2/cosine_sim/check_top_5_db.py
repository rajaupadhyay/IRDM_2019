import sqlite3
from sqlitedict import SqliteDict
import zlib
import pickle
import time
from multiprocessing import Pool
import sys

# batch_1 last claim: 6744
# batch_2 last claim: 204613
# batch_3 last claim: 94843
# batch_4 last claim: 140740

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

# top_5_dict = SqliteDict('top_5_documents_tfidf.sqlite', decode=decompress_set)
# print(top_5_dict[6744])
# print(top_5_dict[204613])
# print(top_5_dict[94843])
# print(top_5_dict[140740])


# idnp_f = open('inverted_doc_name_dict.pickle', 'rb')
# idnp = pickle.load(idnp_f)
# idnp_f.close()
#
# print(idnp['5152616'])



# def test_fn(lst):
#     top_5_dict = SqliteDict('../../top_5_documents_tfidf.sqlite', decode=decompress_set)
#     r = []
#     for ls in lst:
#         r.append(top_5_dict[ls])
#         print(ls)
#         sys.stdout.flush()
#         time.sleep(3)
#
#     top_5_dict.close()
#
#     return r
#
#
#
#
#
#
# def main():
#     args = [75397, 150448, 214861, 156709, 83235]
#
#     pool = Pool(processes=4)
#     print('Processing')
#
#     first_batch_res = pool.apply_async(test_fn, [args])
#     second_batch_res = pool.apply_async(test_fn, [args])
#     third_batch_res = pool.apply_async(test_fn, [args])
#     fourth_batch_res = pool.apply_async(test_fn, [args])
#
#     pool.close()
#     pool.join()
#
#     print(first_batch_res.get())
#     print(second_batch_res.get())
#
#
# if __name__ == '__main__':
#     main()
