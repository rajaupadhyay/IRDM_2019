import pickle
import json
claimToDocsDict_f = open('claimToDocsDict.pickle', 'rb')
claimToDocsDict = pickle.load(claimToDocsDict_f)
claimToDocsDict_f.close()


print(len(claimToDocsDict))


# batch_1 last claim: 25510
# batch_2 last claim: 105886
# batch_3 last claim: 17475
# batch_4 last claim: 219509
