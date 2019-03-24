import pickle
import sqlite3

# inverted_doc_name_dict_f = open('inverted_doc_name_dict.pickle', 'rb')
# inverted_doc_name_dict = pickle.load(inverted_doc_name_dict_f)
# inverted_doc_name_dict_f.close()
#
# # print(inverted_doc_name_dict['597645'])
#
# batch1= [3508468, 4845618, 3541706, 1741662, 1816339]
#
# batch2= [657990, 2979977, 2461570, 1908649, 3664762]
# batch3= [2247943, 2872042, 1724678, 273735, 3488916]
# batch4= [264217, 3023598, 2511050, 1612423, 4849862]
# batch5= [4469627, 304677, 4646988, 2877221, 4252184]
# batch6= [2251341, 2873021, 451816, 4119278, 2161160]
# batch7= [4424082, 747145, 2639111, 4142040, 1000875]
# batch8= [4235595, 3533625, 1457124, 5165955, 609198]
# batch9= [4727442, 4518635, 834098, 2904706, 4015936]
# batch10= [4767495, 998899, 5289509, 4835714, 3948940]
#
# batches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10]
#
# for batch in batches:
#     top_5_names = []
#     for itx in batch:
#         top_5_names.append(inverted_doc_name_dict[str(itx)])
#
#     print(top_5_names)



conn = sqlite3.connect('wiki_corpus.db')
c = conn.cursor()


c.execute('pragma encoding')
print(c.fetchone())

'''
{"id": 155901, "verifiable": "VERIFIABLE", "label": "SUPPORTS", "claim": "Beyonc\u00e9 was given a music award.",
"evidence": [[[179964, 192503, "Beyonce\u0301", 16]], [[179964, 192504, "Beyonce\u0301", 15]]]}
'''

x = 'Beyonce\u00f3'
x = 'Beyonc\u00e9'
print(x)

c.execute('SELECT * FROM wiki WHERE id = ?', (x, ))
# c.execute('SELECT COUNT(*) FROM wiki')

print(c.fetchone())

conn.close()




# conn = sqlite3.connect('test.db')
# c = conn.cursor()
#
# # workers = ProcessPool(processes=4)
#
# count = 0
#
# # c.execute("INSERT INTO dummy VALUES (?,?,?)", ('Beyoncéf', 'testv', '111'))
# x = 'Beyoncé'
# c.execute('SELECT * FROM dummy WHERE id = ?', (x, ))
#
# print(c.fetchone())
# conn.close()
