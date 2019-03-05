import pickle
import numpy as np
import redis
from scipy.sparse import coo_matrix
import time
from collections import defaultdict
import string
from nltk import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from nltk import word_tokenize
from sqlitedict import SqliteDict
import zlib
from numpy import linalg as LA
import pandas as pd

# conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
# print(conn.keys())
# res = conn.hgetall('doc_name_index')
# print(len(res))
# print(type(res))
#
# res = {v: k for k,v in res.items()}
#
# with open('inverted_doc_name_dict.pickle', 'wb') as handle:
#     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


idx_to_name_dict_f = open('inverted_doc_name_dict.pickle', 'rb')
idx_to_name_dict = pickle.load(idx_to_name_dict_f)
idx_to_name_dict_f.close()

print(idx_to_name_dict['3510191'])

lst1 = [3510191, 5149384, 291223, 3541706, 3508468]
lst2 = [541502, 2454098, 640333, 596034, 993333]
lst3 = [526907, 2247943, 5163086, 1919397, 4849139]
lst4 = [264217, 281895, 1612423, 2413013, 2511050]
lst5 = [2305553, 4500037, 2828894, 2835962, 2862482]
lst6 = [1002188, 4717964, 146765, 121193, 141522]
lst7 = [2839403, 2818972, 2829375, 2901253, 672264]
lst8 = [3492651, 725991, 4284319, 2856246, 2068601]
lst9 = [4727442, 4718687, 62350, 4710207, 819893]
lst10 = [1581779, 1414963, 1510567, 1600063, 3550517]

ctr = 1
for lst in [lst1, lst2, lst3, lst4, lst5, lst6, lst7, lst8, lst9, lst10]:
    print('ctr val', ctr)
    ctr += 1
    res = []
    for itx in lst:
        res.append(idx_to_name_dict[str(itx)])

    print(res)
