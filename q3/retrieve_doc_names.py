import pickle
import sqlite3

inverted_doc_name_dict_f = open('inverted_doc_name_dict.pickle', 'rb')
inverted_doc_name_dict = pickle.load(inverted_doc_name_dict_f)
inverted_doc_name_dict_f.close()

# print(inverted_doc_name_dict['597645'])

batch1= [2912923, 3761580, 1100061, 2823548, 4525960]

# batch2= [2206423, 703559, 4441695, 1515196, 4164124]
# batch3= [2821719, 2993350, 2925579, 2056107, 1199268]
# batch4= [2292290, 1196885, 2194262, 614051, 3479931]
# batch5= [2878423, 2937676, 2808351, 5276303, 1863570]
# batch6= [2952216, 409480, 1765202, 2069582, 2939083]
# batch7= [264508, 936341, 865249, 123859, 3853929]
# batch8= [3011270, 2963696, 1385802, 2947329, 2889432]
# batch9= [2817900, 2928246, 3773937, 2929726, 3803191]
# batch10= [2985953, 2965524, 2491648, 338785, 4468568]

# [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10]

# for batch in [batch1]:
#     top_5_names = []
#     for itx in batch:
#         top_5_names.append(inverted_doc_name_dict[str(itx)])
#
#     print(top_5_names)



conn = sqlite3.connect('wiki_corpus.db')
c = conn.cursor()

c.execute('SELECT * FROM wiki WHERE id = ?', ("Since_I've_Been_Loving_You", ))
# c.execute('SELECT COUNT(*) FROM wiki')

print(c.fetchone()[1])
