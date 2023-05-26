"""Scoring script used in the backend for inference"""
import os
import json
import logging
from time import time
from glob import glob
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from torch.nn import Sigmoid
import re
import faiss
import numpy as np
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient, BlobServiceClient
import pathlib


def retrieve_corpora_from_blob():
    """
    Function that will connect to the storage account and download the files specified in the environment variables
    It will then load in a huggingface dataset with the files it has just downloaded and 
    then it will add a faiss index on the embedding column
    """
    credential = ManagedIdentityCredential()
    storage_account = os.getenv("STORAGE_ACCOUNT_NAME")
    storage_container = os.getenv("STORAGE_CONTAINER_NAME")
    folders = os.getenv("FOLDERS")
    folders = folders.split(",")

    corpora = os.getenv("CORPORA")
    corpora = corpora.split(",")

    base_path_local_corpora = "corpora"

    for idx, folder in enumerate(folders):
        corpus = corpora[idx]

        bsc = BlobServiceClient(account_url=f"https://{storage_account}.blob.core.windows.net/", credential=credential)
        cc = bsc.get_container_client(container=storage_container)

        for file_name in cc.list_blob_names(name_starts_with=folder):
            print(f"File_name: {file_name}")
            blob_client = BlobClient(
                account_url=f"https://{storage_account}.blob.core.windows.net/",
                container_name=storage_container,
                blob_name=file_name,
                credential=credential,
            )
            fname = file_name.rsplit(os.path.sep, 1)[-1]
            corpus_file_name = corpus.replace(" ", "")
            p = pathlib.Path(f"{base_path_local_corpora}/{corpus_file_name}/{fname}")
            p.parent.mkdir(parents=True, exist_ok=True)
            print(f"Creating the parents of {p}")

            with p.open("wb") as f:
                blob_contents = blob_client.download_blob()
                blob_contents.readinto(f)

    results = {}
    for corpus_name in corpora:
        corpus_data = load_from_disk(os.path.join(base_path_local_corpora, corpus_name.replace(" ", "")))
        corpus_data.add_faiss_index(column="embedding", metric_type=faiss.METRIC_INNER_PRODUCT)

        results[corpus_name] = corpus_data

    return results


def init():
    """
    Loads the models from the Azure ML model registry
    """
    global bi_encoder
    global cross_encoder
    global corpora

    model_locations = os.getenv("AZUREML_MODEL_DIR")
    model_locations = os.path.join(model_locations, "models")

    bi_encoder_folders = glob(os.path.join(model_locations, "bi_encoder"))
    print("bi_encoder_folders:", bi_encoder_folders)
    bi_encoder = SentenceTransformer(bi_encoder_folders[0])
    bi_encoder.max_seq_length = 512  #Truncate long passages to 256 tokens
    bi_encoder.tokenizer.truncate_sequences = False

    cross_encoder_folders = glob(os.path.join(model_locations, "cross_encoder"))
    print("cross_encoder_folders:", cross_encoder_folders)
    cross_encoder = CrossEncoder(cross_encoder_folders[0])

    corpora = retrieve_corpora_from_blob()


def split_in_sentences(text):
    """
    Split a text in sentences

    :param text: The text to split

    :return: A list of sentences
    """
    regex = '\.{1}\s+[A-Z]|\s+\d{1,2}\.\s{1}'

    prefixes = re.findall(regex, text)

    txt = re.split(regex, text)

    # If there are no splits to make, return OG text
    res = [txt[0]]
    print(len(prefixes))
    for x in range(len(prefixes)):
        prefix = prefixes[x]
        next_text = txt[x + 1]
        print(next_text)

        res[-1] = res[-1] + prefix[0]
        acc = prefix[1:] + next_text

        res.append(acc)

    return res


# given the list of sentences and the list of their scores
# return a list of either strings (in case the sentence does not score well) or (sentence, color) tuples in case it does score well
def get_sentence_color(sentences, scores):
    """
    Get the color of the sentence based on the score

    :param sentences: The list of sentences
    :param scores: The list of scores

    :return: A list of either strings (in case the sentence does not score well) or (sentence, color) tuples in case it does score well
    """
    color_100 = "rgb(0,255,0)"
    color_90 = "rgb(150,255,0)"
    color_75 = "rgb(220,255,100)"
    quantiles = np.quantile(scores, [0.75, 0.9, 1])

    res = []
    for i in range(len(scores)):
        score = scores[i]
        sentence = sentences[i]
        if quantiles[0] <= score < quantiles[1]:
            res.append((sentence, color_75))
        elif quantiles[1] <= score < quantiles[2]:
            res.append((sentence, color_90))
        elif score >= quantiles[2]:
            res.append((sentence, color_100))
        else:
            res.append(sentence)

    return res


def score_sentences(query_encoded, dictionary):
    """
    Score the sentences in the dictionary

    :param query_encoded: The encoded query
    :param dictionary: The dictionary containing the text and the cross_encoder_prediction_score

    """
    text = []
    for i, score in enumerate(dictionary["cross_encoder_prediction_score"]):
        print(f"score: {score}")
        if score > 0.8:
            # print(f"paragraph: {dictionary['text'][i]}")
            sentences = split_in_sentences(dictionary['text'][i])
            # print(f"sentences: {sentences}")
            sentences_encoded = bi_encoder.encode(sentences)
            scores = []
            for i in range(len(sentences_encoded)):
                cos_sim = np.dot(sentences_encoded[i], np.squeeze(query_encoded)
                                 ) / (np.linalg.norm(sentences_encoded[i]) * np.linalg.norm(query_encoded))
                scores.append(cos_sim)
                # print(f"cos sim for {sentences[i]}: {cos_sim}")

            t = get_sentence_color(sentences, scores)
            text.append(t)
        else:
            text.append(dictionary['text'][i])
    d = {"text": text}

    dictionary.update(d)


def cross_encoder_reranking(query, samples, prefix=""):
    """
    Cross-encoder reranking of the samples

    :param query: The query
    :param samples: The samples to rerank
    :param prefix: The prefix to add to the keys of the samples

    :return: The samples with the cross-encoder scores
    """

    def sort_list(indexs, items):
        """
        Sort a list based on the indexs

        :param indexs: The indexs to sort the list
        :param items: The list to sort

        :return: The sorted list
        """
        return [items[i] for i in indexs]

    # paragraphs are empty? -> no results where found so return empty dict
    if len(samples["text"]) != 0:
        cross_input = list(map(lambda x: query + [x], samples['text']))
        cross_scores = cross_encoder.predict(
            cross_input, convert_to_numpy=True, batch_size=64, activation_fct=Sigmoid()
        )

        best_indexes = cross_scores.argsort()[::-1]
        # sort the samples returned from faiss in the order of the cross-encoder + filter out unnecessary columns to return
        res_dict = {prefix + str(k): sort_list(best_indexes, v) for k, v in samples.items() if k not in ["embedding"]}
        res_dict[prefix + 'cross_encoder_reordering'] = best_indexes.tolist()
        res_dict[prefix + 'cross_encoder_prediction_score'] = [float(i) for i in sort_list(best_indexes, cross_scores)]
    else:
        # samples has the same keys as res_dict from the if true case
        samples.pop("embedding", None)
        res_dict = samples
        res_dict[prefix + 'cross_encoder_reordering'] = []
        res_dict[prefix + 'cross_encoder_prediction_score'] = []

    return res_dict


def run(raw_query):
    """
    Evaluate the query and retrieves the results

    :param raw_query: The query to evaluate

    :return: The results
    """
    logging.info(f"Received the raw query{raw_query}")

    js = json.loads(raw_query)
    query = js['query']
    selected_corpus = js['corpus']
    highlighting = js['highlighting']
    top_k_standard = js['top_k']

    encode_start = time()
    query_embedding = bi_encoder.encode(query, convert_to_tensor=False, convert_to_numpy=True)

    # so that there is no error later
    filter_start = filter_end = 0
    encode_end_faiss_start = time()

    _scores, samples = corpora[selected_corpus].get_nearest_examples('embedding', query_embedding, k=top_k_standard)

    faiss_end = time()

    print("num samples", len(samples["text"]))
    cross_start = time()
    res_dict = cross_encoder_reranking(query=query, samples=samples)
    cross_end = time()

    res_dict['bi_encoder_time'] = encode_end_faiss_start - encode_start
    res_dict['faiss_time'] = faiss_end - encode_end_faiss_start - (filter_end-filter_start)
    res_dict['cross_encoder_time'] = cross_end - cross_start
    res_dict['filter_time'] = filter_end - filter_start

    sentences_start = time()

    if highlighting:
        score_sentences(query_encoded=query_embedding, dictionary=res_dict)
    sentences_end = time()
    res_dict['sentence_time'] = sentences_end - sentences_start

    return res_dict
