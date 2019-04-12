import sqlite3
import json
import os
import importlib.util
import re
from multiprocessing import Pool as ProcessPool
from tqdm import tqdm

wiki_files_path = '../data/wiki-pages/wiki-pages/'

def file_reader_generator(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data


def get_contents(file_name):
    documents = []
    with open(wiki_files_path+file_name, 'r') as openfile:
        for line in file_reader_generator(openfile):
            json_dict = json.loads(line)
            file_key = json_dict['id']
            if file_key:
                documents.append((file_key, json_dict['text'], json_dict['lines']))


    return documents

# def main():
#
#     list_of_wiki_files = os.listdir(wiki_files_path)
#     list_of_wiki_files.sort()
#
#     conn = sqlite3.connect('../wiki_corpus.db')
#     c = conn.cursor()
#
#     # workers = ProcessPool(processes=4)
#
#     count = 0
#     with tqdm(total=len(list_of_wiki_files)) as pbar:
#         for fn in list_of_wiki_files:
#             with open(wiki_files_path+fn, 'rb') as openfile:
#                 for line in file_reader_generator(openfile):
#                     json_dict = json.loads(line)
#                     file_key = json_dict['id']
#                     if file_key:
#                         triplets = (file_key, json_dict['text'], json_dict['lines'])
#                         c.execute("INSERT INTO wiki VALUES (?,?,?)", triplets)
#             conn.commit()
#             pbar.update()
#
#     conn.close()
#
#
# if __name__ == '__main__':
#     main()


conn = sqlite3.connect('wiki_corpus.db')
c = conn.cursor()

# c.execute('SELECT * FROM wiki WHERE id = ?', ("Nikolaj_Coster-Waldau", ))
# c.execute('SELECT COUNT(*) FROM wiki')
c.execute('SELECT * FROM wiki WHERE id = ?', ("Tetri", ))

print(c.fetchone()[1])


# c.execute('SELECT * FROM wiki')
#
# ctr = 0
# for row in c:
#     ctr += 1
#     # if ctr == 21:
#     #     break
#
#     lines = row[-1]
#     lines = re.split('\\n\d+\\t', lines)
#     # print(lines)
#     new_lines = []
#     brkr = 0
#     for line in lines[:-1]:
#         actual_line = re.split('\t', line)
#         if brkr == 0:
#             new_lines.append(actual_line[1])
#         else:
#             new_lines.append(actual_line[0])
#
#         brkr += 1
#
#     check_existence = [itx in row[1] for itx in new_lines]
#
#     if not all(check_existence):
#         print(row[0])
#         break
