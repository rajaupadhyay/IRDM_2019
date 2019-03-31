import pickle
import sqlite3

inverted_doc_name_dict_f = open('inverted_doc_name_dict.pickle', 'rb')
inverted_doc_name_dict = pickle.load(inverted_doc_name_dict_f)
inverted_doc_name_dict_f.close()

# print(inverted_doc_name_dict['597645'])

batch1= [2915033, 1736409, 2786806, 3363842, 823236]

batch2= [2979977, 2879214, 2206423, 272010, 703559]
batch3= [2872042, 273735, 2979977, 4798450, 2809139]
batch4= [264217, 2292290, 1196885, 2194262, 614051]
batch5= [2960232, 2251341, 2934343, 2961684, 2979977]
batch6= [76735, 2467399, 2979977, 4518635, 1409069]
batch7= [4767495, 998899, 2935783, 2979977, 1192936]
batch8= [2493781, 1570617, 2929726, 5043000, 5233464]
batch9= [1251050, 2956303, 1283452, 2914444, 2920767]
batch10= [2979977, 2813070, 4754520, 2988580, 2842063]

batches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10]

for batch in batches:
    top_5_names = []
    for itx in batch:
        top_5_names.append(inverted_doc_name_dict[str(itx)])

    print(top_5_names)



# conn = sqlite3.connect('wiki_corpus.db')
# c = conn.cursor()
#
#
# c.execute('pragma encoding')
# print(c.fetchone())
#
# '''
# {"id": 155901, "verifiable": "VERIFIABLE", "label": "SUPPORTS", "claim": "Beyonc\u00e9 was given a music award.",
# "evidence": [[[179964, 192503, "Beyonce\u0301", 16]], [[179964, 192504, "Beyonce\u0301", 15]]]}
# '''
#
# x = 'Beyonce\u00f3'
# x = 'Beyonc\u00e9'
# print(x)
#
# c.execute('SELECT * FROM wiki WHERE id = ?', (x, ))
# # c.execute('SELECT COUNT(*) FROM wiki')
#
# print(c.fetchone())
#
# conn.close()




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
