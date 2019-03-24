from sqlitedict import SqliteDict
import pickle
import zlib
import sqlite3
import json

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

claims_db = SqliteDict('testing_db.sqlite', autocommit=True, encode=compress_set, decode=decompress_set)

def add_key_to_db(key, val_list):
    claims_db[key] = val_list

def file_reader_generator(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data

total_itrs = 0
with open('data/test_batches/dev_set.jsonl', 'r') as openfile:
    for line in file_reader_generator(openfile):

        if total_itrs%50 == 0:
            print(total_itrs)

        total_itrs += 1

        json_dict = json.loads(line)
        claim_id = json_dict['id']
        verifiable = json_dict['verifiable']
        label = json_dict['label']
        evidence = json_dict['evidence']
        claim = json_dict['claim']

        lst = [verifiable, label, claim, evidence]

        add_key_to_db(claim_id, lst)



claims_db.close()
