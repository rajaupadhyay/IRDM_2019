# from BTrees.OOBTree import OOSet
# from BTrees.OOBTree import OOBTree
import sys
from sqlitedict import SqliteDict
import pandas as pd
import zlib, pickle, sqlite3

# t = OOBTree()
#
# for i in range(200):
#     print(i)
#     s = OOSet([x for x in range(1000000)])
#     dict_ = {i: s}
#     t.update(dict_)
#
# print(sys.getsizeof(t))


# import h5py
# h = h5py.File('mytestfile.hdf5', 'r')
#
# cs = h['idx_0']['docs']
# # print(cs)
# print(5 in cs)
#
# for x in range(100000):
#     t = x in cs
#     print(t)


def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))
#
# mydict = SqliteDict('my_db.sqlite', autocommit=True, encode=compress_set, decode=decompress_set)
#
# for i in range(200):
#     print(i)
#     mydict['idx_{}'.format(i)] = {x for x in range(1000000)}
#
# mydict.close()


# df = pd.read_csv('words_in_docs.txt', sep=' ', header=None)
# df.columns = ["n", "word", "a", "i", "doc_freq", "d", "b", "c", "id", "idx"]
#
# df.to_csv('high_freq_words_stash.csv', sep=',')

with SqliteDict('high_frequency_stash.sqlite', decode=decompress_set) as mydict:
    res = mydict['ividx_karlıköi']
    r3 = mydict['ividx_leonidovitch']
    rr = res.union(r3)

    print(len(res), len(r3), len(rr))
