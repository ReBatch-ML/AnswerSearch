""" Scoring script of the semantic search backend"""
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
import pandas as pd
import math
from difflib import SequenceMatcher
import torch
from difflib import SequenceMatcher
from typing import Dict

from typing import List


class Context():
    """
    Context class to store the context of a search result
    """

    def __init__(self, paragraph_idx: int, idxs_before: List[int], idxs_after: List[int], text: str):
        self.text = text
        self.paragraph_idx = paragraph_idx
        self.idxs_before = idxs_before
        self.idxs_after = idxs_after
        self.duplicate = False

    def __eq__(self, other):
        """
        Check if 2 contexts are similar enough to be considered duplicates. If they have the exact same paragraph ids that make up
        the entire context, they are considered duplicates.

        Args:
            other (Context): context to compare to

        Returns:
            bool: True if considered duplicates, False otherwise
        """
        if isinstance(other, Context):
            all_idxs_self = self.idxs_before + [self.paragraph_idx] + self.idxs_after
            all_idxs_other = other.idxs_before + [other.paragraph_idx] + other.idxs_after

            # sort just to be sure as [1,2,3] != [1,3,2] but should be equal in our case
            all_idxs_self.sort()
            all_idxs_other.sort()
            return all_idxs_self == all_idxs_other
        return False

    def __str__(self) -> str:
        """
        Convert the context to a string so it can be printed in a UI

        Returns:
            str: The text that makes up the context, or the word DUPLICATE if the context is a duplicate
        """
        return self.text if not self.duplicate else "DUPLICATE"


def init():
    """
    Initializes the backend API by loading the model and corpora
    """
    global bi_encoder
    global cross_encoder
    global corpus_embeddings
    global staff_reg_officials_embeddings
    global staff_reg_others_embeddings
    # global device

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_locations = os.getenv("AZUREML_MODEL_DIR")
    model_locations = os.path.join(model_locations, "models")
    bi_encoder_folders = glob(os.path.join(model_locations, "bi_encoder"))
    print("bi_encoder_folders:", bi_encoder_folders)
    bi_encoder = SentenceTransformer(bi_encoder_folders[0])
    # bi_encoder = SentenceTransformer("models/bi_encoder")
    bi_encoder.max_seq_length = 512  #Truncate long passages to 256 tokens
    bi_encoder.tokenizer.truncate_sequences = False
    # bi_encoder = bi_encoder.to(device)

    cross_encoder_folders = glob(os.path.join(model_locations, "cross_encoder"))
    print("cross_encoder_folders:", cross_encoder_folders)
    cross_encoder = CrossEncoder(cross_encoder_folders[0])
    # cross_encoder = cross_encoder.to(device)

    connection_string = os.getenv("CONNECTION_STRING")
    print("CONNECTION_STRING:", connection_string)

    location_public_reg = "/corpora/public_registry"
    location_staff_reg_officials = "/corpora/stafreg_registry_officials"
    location_staff_reg_others = "/corpora/stafreg_registry_others"

    # dataset is small enough for a flat index
    staff_reg_officials_embeddings = load_from_disk(location_staff_reg_officials)
    staff_reg_officials_embeddings.add_faiss_index(column='embeddings', metric_type=faiss.METRIC_INNER_PRODUCT)
    # staff_reg_officials_embeddings.load_faiss_index('embeddings', os.path.join(location_staff_reg_officials, "IVF76_Flat.faiss"))

    # dataset is small enough for a flat index
    staff_reg_others_embeddings = load_from_disk(location_staff_reg_others)
    staff_reg_others_embeddings.add_faiss_index(column='embeddings', metric_type=faiss.METRIC_INNER_PRODUCT)
    # staff_reg_others_embeddings.load_faiss_index('embeddings', os.path.join(location_staff_reg_others, "IVF47_Flat.faiss"))

    corpus_embeddings = load_from_disk(location_public_reg)
    corpus_embeddings.add_faiss_index(column='embeddings', metric_type=faiss.METRIC_INNER_PRODUCT)


def __remove_text_overlap(texts: List[str]):
    """
    Removes the overlap between the different chunks of the text.

    Args:
        texts (List[str]): List of all the chunks of the text in chronological order

    Returns:
        List[str]: List the chunks of the text in chronological order with the overlap removed
    """
    t = [texts[0]]
    for i in range(1, len(texts)):
        # last 150 characters of previous text and first 150 characters of current text
        previous_text = t[i - 1][-150:]
        current_text = texts[i][:150]

        sm = SequenceMatcher(None, previous_text, current_text)
        a, b, s = sm.find_longest_match(0, len(previous_text), 0, len(current_text))

        # overlapping length is long enough and the overlap is at the beginning of the current text and the end of the previous text
        if s > 19 and b == 0 and a == len(previous_text) - s:
            removed_overlap = texts[i][s:]
            t.append(removed_overlap)
        else:
            print("THIS SHOULD NOT HAPPEN: THERE IS NO OVERLAP BETWEEN 2 DIFFERENT CHUNKS OF THE TEXT FOR SOME REASON")
    return t


def __connect_overlapping_chunks(context_dict: Dict, semsearch_paragraph_index: int):
    """
    Connects the different chunks of the text together and keep track of the idxs of the different chunks that make up the context.

    Args:
        context_dict (Dict): Information about the different paragraphs that make up the context such as the idxs of all the paragraphs and the text of the paragraphs
        semsearch_paragraph_index (int): The idx that corresponds to the semsearch paragraph inside the lists of the context_dict

    Returns:
        Context: Context object that contains the text of the context, the idxs of the different paragraphs that make up the context and the idx of the semsearch paragraph
    """
    tokenizer = bi_encoder.tokenizer
    texts = context_dict["texts"]
    t = __remove_text_overlap(texts)

    # truncate chunks to less than 1800 tokens with the idx in the middle
    text_before = " ".join(t[:semsearch_paragraph_index])
    text_paragraph = t[semsearch_paragraph_index]
    text_after = " ".join(t[semsearch_paragraph_index + 1:])

    # keep track of the idxs of the different chunks
    idxs_before = context_dict["idxs"][:semsearch_paragraph_index]
    idxs_paragraph = context_dict["idxs"][semsearch_paragraph_index]
    idxs_after = context_dict["idxs"][semsearch_paragraph_index + 1:]

    # make sure we splitted the idxs correctly
    assert idxs_paragraph == context_dict["paragraph_idx"]

    # tokenize the different paragraphs (without overlap)
    tokens = tokenizer(t)["input_ids"]
    tokens_per_chunk_before = tokens[:semsearch_paragraph_index]
    tokens_paragraph = tokens[semsearch_paragraph_index]
    tokens_per_chunk_after = tokens[semsearch_paragraph_index + 1:]

    # get the total amount of tokens before and after the paragraph that was returned by semantic search
    amount_tokens_before = sum((len(tokens) for tokens in tokens_per_chunk_before))
    amount_tokens_after = sum((len(tokens) for tokens in tokens_per_chunk_after))

    # if the total size of the context is less than 1800 tokens, we can keep it all
    if amount_tokens_before + len(tokens_paragraph) + amount_tokens_after < 1800:
        full_text = " ".join([text_before, text_paragraph, text_after])
        c = Context(paragraph_idx=idxs_paragraph, idxs_before=idxs_before, idxs_after=idxs_after, text=full_text)

        return c
    else:
        # truncate the context before and after the paragraph (the paragraph is always kept as is)
        amount_tokens_context_tokens = 1800 - len(tokens_paragraph)
        # ideal amount of tokens per side
        balanced_amount_tokens_per_side = amount_tokens_context_tokens // 2

        # Incrementally add tokens to the context until we reach the desired amount of tokens
        # This way we can keep track of which paragraphs we added to the context and which ones are completely discarded
        total_tokens = 0
        accepted_tokens_before = []
        accepted_tokens_after = []
        accepted_idxs_before = []
        accepted_idxs_after = []

        # Add chunks to the front and back simulataneously until we reach the desired amount of tokens
        while total_tokens < amount_tokens_context_tokens:
            if len(tokens_per_chunk_before) > 0:
                accepted_tokens_before = tokens_per_chunk_before.pop() + accepted_tokens_before
                accepted_idxs_before = [idxs_before.pop()] + accepted_idxs_before

            if len(tokens_per_chunk_after) > 0:
                accepted_tokens_after += tokens_per_chunk_after.pop(0)
                accepted_idxs_after.append(idxs_after.pop(0))
            total_tokens = len(accepted_tokens_before) + len(accepted_tokens_after)

        # paragraphs are big so we might have added too many tokens, so we need to remove some of the last paragraph in the front and back
        reduced_tokens_before = accepted_tokens_before[-balanced_amount_tokens_per_side:]
        reduced_tokens_after = accepted_tokens_after[:balanced_amount_tokens_per_side]

        # One of the sides might be longer than the other (for example when the paragraph was located at the bottom of the complete context),
        # so we need to add tokens to the other side to make sure we completely fill the context
        extra_needed_before = balanced_amount_tokens_per_side - len(reduced_tokens_after)
        extra_needed_after = balanced_amount_tokens_per_side - len(reduced_tokens_before)

        tokens_before = accepted_tokens_before[-balanced_amount_tokens_per_side - extra_needed_before:]
        tokens_after = accepted_tokens_after[:balanced_amount_tokens_per_side + extra_needed_after]

        # decode the tokens back to text
        # (the first and last paragraph might be truncated so we need to decode so that we know which text exactly is in the context)
        decoded_text_before = tokenizer.decode(tokens_before, skip_special_tokens=True)
        decoded_text_after = tokenizer.decode(tokens_after, skip_special_tokens=True)

        full_text = " ".join([decoded_text_before, t[semsearch_paragraph_index], decoded_text_after])

        c = Context(
            paragraph_idx=idxs_paragraph,
            idxs_before=accepted_idxs_before,
            idxs_after=accepted_idxs_after,
            text=full_text
        )

        return c


def expand_context(ds, og_samples):
    """
    Function that expands paragraphs to a context. It will look for all paragraphs that have the same chunk_id and pass all these paragraphs to the
    __connect_overlapping_chunks function.

    Args:
        ds (Dataset): Huggingface Dataset that has all the paragraphs. It should have a column called "chunk_id" that indicates which chunk a paragraph belongs to. 
        og_samples (Dict): Result returned by the semantic search
        
    Returns:
        Dict: Updates samples dict with a new column called "context" that contains the expanded context
    """
    samples = og_samples.copy()
    context = []
    for i in range(len(samples["idx"])):
        if samples["chunk_id"][i] != -1:
            chunk_id = samples["chunk_id"][i]

            look_ahead_idx = samples["idx"][i] + 1
            while ds[look_ahead_idx]["chunk_id"] == chunk_id:
                look_ahead_idx += 1

            context_dict = {
                "texts": [ds[j]['text'] for j in range(chunk_id, look_ahead_idx)],
                "idxs": [ds[j]['idx'] for j in range(chunk_id, look_ahead_idx)],
                "paragraph_idx": samples["idx"][i],
            }

            assert context_dict["paragraph_idx"] == context_dict["idxs"][samples["idx"][i] - chunk_id]

            ctx = __connect_overlapping_chunks(context_dict, samples["idx"][i] - chunk_id)

            context.append(ctx)

        else:
            c = Context(paragraph_idx=samples["idx"][i], idxs_before=None, idxs_after=None, text=samples["text"][i])
            context.append(c)

    samples["context"] = context

    # find all samples that have expended contexts
    chunked_sample_idxs = [i for i, c_id in enumerate(samples["chunk_id"]) if c_id != -1]
    print("Chunked samples: ", chunked_sample_idxs)
    # first one can't be a duplicate
    for num, i in enumerate(chunked_sample_idxs[1:], 1):
        chunk_id = samples["chunk_id"][i]
        # compare to all previous samples
        for j in chunked_sample_idxs[:num]:
            ci = samples["context"][i]
            cj = samples["context"][j]

            if ci == cj:
                print(f"{i} is a duplicate of {j}")
                ci.duplicate = True
                break

    samples["context"] = [str(c) for c in samples["context"]]
    return samples


def split_in_sentences(text):
    """
    Split the paragrpahs in sentences so we can evaluate similarity on sentence level

    Args:
        text (str): paragraph text

    Returns:
        List[str]: List of the sentences in the given paragraph
    """
    regex = '\.{1} ?[A-Z]|•|\n|[ +(-][0-9]{1,2}[)-] '

    prefixes = re.findall(regex, text)

    txt = re.split(regex, text)

    # If there are no splits to make, return OG text
    res = [txt[0]]
    for x in range(len(prefixes)):
        prefix = prefixes[x]
        next_text = txt[x + 1]

        if prefix.startswith('.'):
            # add the . add the end of the previous sentence
            res[-1] = res[-1] + '.'
            # place the capital letter back in front of the sentence
            acc = prefix[-1] + next_text
        elif prefix != '\n':
            acc = prefix + next_text
        else:
            acc = next_text
        res.append(acc)

    return res


def get_sentence_color(sentences, scores):
    """
    given the list of sentences and the list of their scores
    return a list of either strings (in case the sentence does not score well) or (sentence, color) tuples in case it does score well
    
    Args:
        sentences (List[str]): List of sentences belonging to 1 paragraph
        scores (List[float]): Similarity scores of each sentence to the query

    Returns:
        List[Tuple(str,str)]: List of either a Tuple (sentence, color string) or just string of the confidence isn't high enough for highlighting
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
    Split sentences of highly confident paragraphs and perform cos sim with each sentence and the query.
    Afterwards give color information to relevant sentences.

    Args:
        query_encoded (np.array): embedding of the query
        dictionary (Dict): Result of the semantic search before highlighting. Will be updated in place to have the result with highlighting
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


def filter_retrieved_ex(retrieved_examples, years, keep=32):
    """
    Filter the results based on publishing year. If there's not enough documents over to keep, then take unfiltered samples too.

    Args:
        retrieved_examples (Dict): Retrieved items from the faiss index
        years (List[int]): List of the acceptible years.
        keep (int, optional): Amount of results to show or pad to when the filter is too strict. Defaults to 32.

    Returns:
        Tuple(Dict, Dict): Tuple of result dict with only filtered items and of padding items that are outside the filter (if needed)
    """
    retex_df = pd.DataFrame(retrieved_examples)
    retex_df['year'] = retex_df['doc_id'].apply(lambda x: int(x.split(' ')[2]))

    filtered_df = retex_df[retex_df.year.isin(years)].copy()

    filtered_df.drop(columns=['year'], inplace=True)
    filtered_df = filtered_df[:keep]
    num_extra_results = abs(len(filtered_df) - keep)
    print(f"Number of filtered results {len(filtered_df)}")
    print(f"Extra results to add from other years is {num_extra_results}")
    if num_extra_results > 0:
        print(f'Warning: lacking {num_extra_results} results after filtering')
        others_df = retex_df[~retex_df.year.isin(years)]
        others_df = others_df[:num_extra_results]
        others_dict = others_df.to_dict(orient='list')
    else:
        others_dict = None
    result_dict = filtered_df.to_dict(orient='list')

    return result_dict, others_dict


def cross_encoder_reranking(query, samples, prefix=""):
    """
    Use the cross encoder to rerank the samples retrieved by the faiss index.

    Args:
        query (List[str]): the query sentence in text.
        samples (Dict): Samples retrieved from a faiss index
        prefix (str, optional): Normally this is empty but in case we have samples from outside of a year filter we give them a prefix. Defaults to "".
        
    Returns:
        Dict: reordered dict from samples
    """

    def sort_list(indexs, items):
        """
        Resort a list based on a list of indexes. On place 0 we place the indexs[0]th item of the items list.

        Args:
            indexs (List[int]): List of where to find each item in the items list.
            items (List[Object]): List of all the original items.

        Returns:
            List[Object]: Sorted list. 
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
        res_dict = {prefix + str(k): sort_list(best_indexes, v) for k, v in samples.items() if k not in ["embeddings"]}
        res_dict[prefix + 'cross_encoder_reordering'] = best_indexes.tolist()
        res_dict[prefix + 'cross_encoder_prediction_score'] = [float(i) for i in sort_list(best_indexes, cross_scores)]
    else:
        # samples has the same keys as res_dict from the if true case
        samples.pop("embeddings", None)
        res_dict = samples
        res_dict[prefix + 'cross_encoder_reordering'] = []
        res_dict[prefix + 'cross_encoder_prediction_score'] = []

    return res_dict


def run(raw_query):
    """
    Function that is called when the api receives a request

    Args:
        raw_query (JSON string): Request body that needs to be parsed.

    Returns:
        Dict: Result of the semantic search.
    """
    logging.info(f"Received the raw query{raw_query}")

    js = json.loads(raw_query)
    query = js['query']
    corpus = js['corpus']
    highlighting = js['highlighting']
    top_k_standard = js['top_k']

    encode_start = time()
    query_embedding = bi_encoder.encode(query, convert_to_tensor=False, convert_to_numpy=True)

    # so that there is no error later
    filter_start = filter_end = 0
    encode_end_faiss_start = time()

    samples_extra = None
    if corpus == "Public Registry":
        years = js['years']
        filtering = years != list(range(1993, 2023))
        # add more samples in case of filtering
        top_k_extended = top_k_standard if not filtering else top_k_standard * 6
        _scores, samples = corpus_embeddings.get_nearest_examples('embeddings', query_embedding, k=top_k_extended)

        filter_start = time()
        if filtering:
            samples, samples_extra = filter_retrieved_ex(retrieved_examples=samples, years=years, keep=top_k_standard)
        filter_end = time()
    elif corpus == "Staff Regulations (EU officials)":
        _scores, samples = staff_reg_officials_embeddings.get_nearest_examples(
            'embeddings', query_embedding, k=top_k_standard
        )
    elif corpus == "Staff Regulations (other servants)":
        _scores, samples = staff_reg_others_embeddings.get_nearest_examples(
            'embeddings', query_embedding, k=top_k_standard
        )
    faiss_end = time()

    print("num samples", len(samples["text"]))
    cross_start = time()
    res_dict = cross_encoder_reranking(query=query, samples=samples)
    if len(samples["text"]) == 0:
        print("THERE ARE NO RESULTS FROM THAT YEAR. WE CAN ONLY RETURN RESULTS FROM OTHER YEARS!!!!!!!" * 10)

    if samples_extra is not None:
        res_dict_extra = cross_encoder_reranking(query=query, samples=samples_extra, prefix="extra_")
        # add the extra items to res_dict. There should be no overlapping keys because of the prefix difference
        res_dict.update(res_dict_extra)
    cross_end = time()

    context_start = time()
    if corpus == "Public Registry":
        res_dict = expand_context(ds=corpus_embeddings, og_samples=res_dict)
    context_end = time()

    sentences_start = time()
    if highlighting:
        score_sentences(query_encoded=query_embedding, dictionary=res_dict)
    sentences_end = time()

    # add time information
    res_dict['bi_encoder_time'] = encode_end_faiss_start - encode_start
    res_dict['faiss_time'] = faiss_end - encode_end_faiss_start - (filter_end-filter_start)
    res_dict['cross_encoder_time'] = cross_end - cross_start
    res_dict['filter_time'] = filter_end - filter_start
    res_dict['sentence_time'] = sentences_end - sentences_start
    res_dict['context_time'] = context_end - context_start

    return res_dict


# if __name__ == "__main__":
#     init()
#     q = json.dumps({'query':["Sanctions against Russia"]})
#     res = run(q)

#     os.environ["AZUREML_MODEL_DIR"] = "."

#     with open("test.json", 'w') as f:
#         json.dump(res, f)
