import pickle

inverted_doc_name_dict_f = open('inverted_doc_name_dict.pickle', 'rb')
inverted_doc_name_dict = pickle.load(inverted_doc_name_dict_f)
inverted_doc_name_dict_f.close()

# print(inverted_doc_name_dict['597645'])

batch1= [2247943]
batch2= [2251341]
batch3= [721047, 4633511, 747145, 5245815, 2639111]
batch4= [4235595]
batch5= [2904706, 834098, 2582840, 4015936]
batch6= [2659007, 4831625, 4835714, 4774169, 1327510]
# batch7= [2244044, 2247615, 4653429, 115387, 5022563]
# batch8= [1081459, 1455225, 1768104, 1784834, 1900449]
# batch9= [4718687, 3924480, 2177300, 4825456, 4778926]
# batch10= [4740841, 1581779, 4957710, 4681352, 1493822]

for batch in [batch1, batch2, batch3, batch4, batch5, batch6]:
    top_5_names = []
    for itx in batch:
        top_5_names.append(inverted_doc_name_dict[str(itx)])

    print(top_5_names)
