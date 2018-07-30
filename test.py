# You need to write down your own code here
# Task: Given any head entity name (e.g. Q30) and relation name (e.g. P36), you need to output the top 10 closest tail entity names.
# File entity2vec.vec and relation2vec.vec are 50-dimensional entity and relation embeddings.
# If you use the embeddings learned from Problem 1, you will get extra credits.
#entity2vec.txt and #relation2vec.txt
import numpy as np
import tensorflow as tf
import os
import time
import sys

entity_txt = open("data/entity2id.txt", "r")
#r = open("data/relation2id.txt", "r")
entity_vec = open("data/entity2vec.vec","r")
#print entity_txt


print ("Welcome to Embeding Annalzer Script")

data = "data"

L1=True
def read_vec_by_id(file):
    e = {}
    f = open("%s/%s2id.txt" % (data, file), "r").readlines()
    g = open("%s/%s2vec.vec" % (data, file), "r").readlines()
    f.pop(0)
    for k in f:
        i, r = k.split()
        v = [float(k) for k in g[int(r)].split()]
        e[i] = np.asarray(v)
    return e


def search(vec, entity, return_n=10):
    distance = {}
    for k, v in entity.items():
        if L1:
            distance[k] = np.linalg.norm(vec - v, ord=1)
        else:
            distance[k] = np.linalg.norm(vec - v, ord=2)
    distance = sorted(distance.items(), key=lambda k: k[1])
    return distance[:return_n]


def main():
    entity = read_vec_by_id("entity")
    relation = read_vec_by_id("relation")
    entity_id = "Q30"
    relation_id = "P36"
    tail_vec = entity[entity_id] + relation[relation_id]
    tail_ids = search(tail_vec, entity)
    for i, k in enumerate(tail_ids):
        print("%0d Tail: %6s Score: %.4f" % (i + 1, k[0], k[1]))
        
    print ("******************************************************")    
    entity_id1 = "Q30"
    entity_id2= "Q49"
    rel_vec = entity[entity_id1] - entity[entity_id2]
    rel_ids = search(rel_vec, relation)
    for i, k in enumerate(rel_ids):
        print("%0d Tail: %6s Score: %.4f" % (i + 1, k[0], k[1]))


     
if __name__ == "__main__":
    main()
