"""help functions for milvus database
"""
import numpy as np
import struct
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, Milvus
from pymilvus import connections
import pandas as pd
from pymilvus import utility
import multiprocessing, logging
import time
import json
import gc
import random
import re
import csv
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer



def create_collection():
    """create milvus collection

    Returns:
        _type_: collection
    """
    vector_id = FieldSchema(name="id", dtype=DataType.INT64, descrition="primary field", is_primary=True, auto_id=False)
    vector = FieldSchema(
    name="vector",
    dtype=DataType.FLOAT_VECTOR,
    dim=768
    )
    text = FieldSchema(
    name="text",
    dtype=DataType.VARCHAR,
    max_length=4096,
    )
    chunk_id = FieldSchema(
    name="chunk_id",
    dtype=DataType.INT64,
    )
    doc_id = FieldSchema(
    name="doc_id",
    dtype=DataType.VARCHAR,
    max_length=128,
    )
    v_id = FieldSchema(
    name="v_id",
    dtype=DataType.VARCHAR,
    max_length=128,
    )
    
    doc_title = FieldSchema(
    name="doc_title",
    dtype=DataType.VARCHAR,
    max_length=2048,
    )
    year = FieldSchema(
    name="year",
    dtype=DataType.INT64,
    )
    schema = CollectionSchema(
    fields=[vector_id,vector, text, chunk_id ,v_id,doc_id, doc_title, year],
    description="Vector similarity search"
    )
    collection_name = "milvus_vectors"
    collection = Collection(
    name=collection_name,
    schema=schema,
    using='default',
    shards_num=8,
    auto_id=True
    )
    ivf_index = {"index_type": "IVF_FLAT", "params": {"nlist": 1024}, "metric_type": "IP"}
    hnsw_index = {"index_type": "HNSW", "params": {"M": 64, "efConstruction": 512}, "metric_type": "IP"}
    flat_index = {"index_type": "FLAT", "params": {}, "metric_type": "IP"}
    collection.create_index(field_name="vector", index_params=hnsw_index)
    #collection.load()
    return collection

def connect_to_db():
    """connect to milvus database"""
    connections.connect(
    alias="default",
    user='username',
    password='password',
    host='20.50.24.59',
    port='19530'
    )


def get_start_index():
    """get the start index of the vectors to be inserted

    Returns:
        _type_: number of vectors in collection
    """
    collection = Collection("milvus_vectors") 
    print("number of vectors in collection so far: ", collection.num_entities)
    return collection.num_entities

def load_vectors():
    """load vectors from disk"""
    last = False
    corpus_embeddings = load_from_disk("pub")
    while(not last):
        
        s_idx = get_start_index()

        last, full_vec = select_from_corpus(s_idx,corpus_embeddings)
        collection = Collection("milvus_vectors")      # Get an existing collection.
        try:
            collection.insert(full_vec,insert_data=True) 
            collection.flush()
            #collection.release()
            gc.collect()
            print("inserted vectors: ", s_idx, " to ", s_idx+10000)
        except:
            collection.release()
            gc.collect()
            print("error in inserting vectors")
            time.sleep(10)

def change_index():
    """change index of collection"""
    collection = Collection("milvus_vectors")      # Get an existing collection.
    collection.load()
    collection.release()
    ivf_index = {"index_type": "IVF_SQ8", "params": {"nlist": 1024}, "metric_type": "IP"}
    hnsw_index = {"metric_type":"IP","index_type":"HNSW","params":{"M": 64, "efConstruction": 512}}
    flat_index = {"index_type": "FLAT", "params": {}, "metric_type": "IP"}
    collection.drop_index()
    collection.create_index(
    field_name="vector", 
    index_params=ivf_index
    )
    print("index changed")

        
def search(target, collection):
    """search for a vector in milvus collection

    Args:
        target (_type_): target vector
        collection (_type_): collection to search in

    Returns:
        _type_: _results and time taken to search
    """
   
    search_params = {"metric_type": "IP", "params": {"nprobe":512}, "offset": 0}
    start = time.time()
    results = collection.search(
        data=[target], 
        anns_field="vector", 
        param=search_params,
        output_fields = ["v_id"],
        limit=128, 
        consistecy_level="Strong"
    )
    end = time.time()
    print("Time taken to search 1 vector", end-start)
    hits = results[0]
    
    return hits, end-start    
        

    #collection.release()

def select_from_corpus(start, corpus_embeddings):

    """select vectors from corpus to be inserted

    Returns:
        _type_: whether it is the last batch of vectors, and the vectors to be inserted
    """
    
    end = np.shape(corpus_embeddings)[0]
    last = False
    endv = start+10000
    if end-start<10000:
        print("last vectors")
        last = True
        endv = end

    full_vec = corpus_embeddings[start:endv]
    years = np.array(full_vec["doc_id"])
    #pattern = r"\b\d{4}\b"
    years = [int(x.split()[2]) for x in years]
    return last, [full_vec["idx"], full_vec["embeddings"], full_vec["text"], full_vec["chunk_id"], full_vec["_id"],full_vec["doc_id"], full_vec["title"],years]

def calculate_recall():
    """calculate recall and average position of the search results"""
    bi_encoder = SentenceTransformer('bi_encoder')
    collection = Collection("milvus_vectors")      # Get an existing collection.
    collection.load()

    with open('test_set.csv') as csvfile:
        test_questions = csv.reader(csvfile, delimiter=',')
        row_count = 0
        found_list = []
        av_time = 0
        for row in test_questions:
            found, time = query_text(bi_encoder, row[2],row[1], collection)
            found_list.append(found)
            row_count+=1
            av_time+=time 
        found_count = sum([1 for found in found_list if found!=-1])
        recall = found_count/row_count
        average_position = sum([found for found in found_list if found!=-1])/found_count
        print("recall: ", recall)
        print("average position: ", average_position)
        print("average time taken to search 1 vector: ", av_time/row_count)
    collection.release()

def query_text(bi_encoder, query, target, collection):

    """search for a text in milvus collection

    Returns:
        _type_: position of the target vector in the search results and time taken to search
    """

    query_embedding = bi_encoder.encode(query, convert_to_tensor=False, convert_to_numpy=True)
    

    res, time = search(query_embedding, collection)
    for i in range(len(res)):
        if res[i].entity.get('v_id')==target:
            print("found")
            print("position: ", i)
            return i, time
    return -1, time
    

if __name__ == '__main__':
    connect_to_db()
    #change_index()
    #calculate_recall()
    #utility.drop_collection("milvus_test")
    #create_collection()
    #change_index()
    #load_vectors()
    collection = Collection("milvus_vectors")      # Get an existing collection.
    
    print(collection.num_entities)
    #collection.load()
    #print("collection loaded")
    #collection.release()



    connections.disconnect("default")
