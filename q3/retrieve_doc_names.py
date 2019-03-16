import pickle
import sqlite3

inverted_doc_name_dict_f = open('inverted_doc_name_dict.pickle', 'rb')
inverted_doc_name_dict = pickle.load(inverted_doc_name_dict_f)
inverted_doc_name_dict_f.close()

# print(inverted_doc_name_dict['597645'])

batch1= [3541706, 3508468, 4845618, 5149384, 291223]

batch2= [657990, 777906, 2456273, 1934980, 3175162]
batch3= [2247943, 1724678, 4514583, 4849139, 441063]
batch4= [1612423, 348093, 264217, 2511050, 196483]
batch5= [4469627, 4646988, 2877221, 1080183, 4767363]
batch6= [2873021, 2251341, 2169221, 2175161, 2232762]
batch7= [5324623, 721047, 1897266, 2173709, 646266]
batch8= [3533625, 4235595, 725991, 2856246, 3823817]
batch9=[4727442, 62350, 819893, 108378, 110863]
batch10= [2659007, 4831625, 4835714, 4774169, 1327510]

batches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10]

for batch in batches:
    top_5_names = []
    for itx in batch:
        top_5_names.append(inverted_doc_name_dict[str(itx)])

    print(top_5_names)



# conn = sqlite3.connect('wiki_corpus.db')
# c = conn.cursor()
#
# c.execute('SELECT * FROM wiki WHERE id = ?', ("Since_I've_Been_Loving_You", ))
# # c.execute('SELECT COUNT(*) FROM wiki')
#
# print(c.fetchone()[1])
