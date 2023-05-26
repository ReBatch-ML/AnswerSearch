import torch
from beir.datasets.data_loader import GenericDataLoader
#from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
import os
import argparse
from beir.util import write_to_json, write_to_tsv
from typing import Dict
from tqdm.autonotebook import trange
import logging, os

logger = logging.getLogger(__name__)


class QueryGenerator:

    def __init__(self, model, **kwargs):
        self.model = model
        self.qrels = {}
        self.queries = {}

    @staticmethod
    def save(output_dir: str, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], prefix: str):

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, prefix + "-qrels"), exist_ok=True)

        query_file = os.path.join(output_dir, prefix + "-queries.jsonl")
        qrels_file = os.path.join(output_dir, prefix + "-qrels", "train.tsv")

        logger.info("Saving Generated Queries to {}".format(query_file))
        write_to_json(output_file=query_file, data=queries)

        logger.info("Saving Generated Qrels to {}".format(qrels_file))
        write_to_tsv(output_file=qrels_file, data=qrels)

    def generate(
        self,
        corpus: Dict[str, Dict[str, str]],
        output_dir: str,
        top_p: int = 0.95,
        top_k: int = 25,
        max_length: int = 64,
        ques_per_passage: int = 1,
        prefix: str = "gen",
        batch_size: int = 32,
        save: bool = True,
        save_after: int = 100000,
        beam_search: bool = False
    ):

        logger.info(
            "Starting to Generate {} Questions Per Passage using top-p (nucleus) sampling...".format(ques_per_passage)
        )
        logger.info("Params: top_p = {}".format(top_p))
        logger.info("Params: top_k = {}".format(top_k))
        logger.info("Params: max_length = {}".format(max_length))
        logger.info("Params: ques_per_passage = {}".format(ques_per_passage))
        logger.info("Params: batch size = {}".format(batch_size))

        count = 0
        corpus_ids = list(corpus.keys())
        corpus = [corpus[doc_id] for doc_id in corpus_ids]

        for start_idx in trange(0, len(corpus), batch_size, desc='pas'):

            size = len(corpus[start_idx:start_idx + batch_size])

            if beam_search:
                docs = [text["text"] for text in corpus[start_idx:start_idx + batch_size]]
                encodings = self.model.tokenizer(docs, padding=True, truncation=True, return_tensors="pt")

                with torch.no_grad():
                    # Here we use top_k / top_k random sampling. It generates more diverse queries, but of lower quality
                    # sampling_outputs = self.model.model.generate(
                    #     input_ids=encodings['input_ids'].to(self.model.device),
                    #     max_length=max_length,
                    #     do_sample=True,
                    #     top_p=top_p,
                    #     top_k=top_k,
                    #     num_return_sequences=ques_per_passage
                    # )
                    #
                    # sampling_output = [self.model.tokenizer.decode(output, skip_special_tokens=True) for output in sampling_outputs]

                    smarter_outputs = self.model.model.generate(
                        input_ids=encodings['input_ids'].to(self.model.device),
                        max_length=max_length,
                        num_beams=20,
                        no_repeat_ngram_size=3,
                        num_return_sequences=ques_per_passage,
                        early_stopping=True
                    )

                    smarter_output = [
                        self.model.tokenizer.decode(output, skip_special_tokens=True) for output in smarter_outputs
                    ]
                    #
                    # for (doc_ix, doc) in enumerate(docs):
                    #     print("Doc: ", doc)
                    #
                    #     for generated_ix in range(ques_per_passage):
                    #         print("Sampling Output: ", sampling_output[doc_ix * ques_per_passage + generated_ix])
                    #         print("Smarter Output: ", smarter_output[doc_ix * ques_per_passage + generated_ix])
                    #         print()
                    #
                    #     print("-------------")

                    queries = smarter_output
            else:
                queries = self.model.generate(
                    corpus=corpus[start_idx:start_idx + batch_size],
                    ques_per_passage=ques_per_passage,
                    max_length=max_length,
                    top_p=top_p,
                    top_k=top_k
                )

            assert len(queries) == size * ques_per_passage

            for idx in range(size):
                # Saving generated questions after every "save_after" corpus ids
                if (len(self.queries) % save_after == 0 and len(self.queries) >= save_after):
                    logger.info("Saving {} Generated Queries...".format(len(self.queries)))
                    self.save(output_dir, self.queries, self.qrels, prefix)

                corpus_id = corpus_ids[start_idx + idx]
                start_id = idx * ques_per_passage
                end_id = start_id + ques_per_passage
                query_set = set([q.strip() for q in queries[start_id:end_id]])

                for query in query_set:
                    count += 1
                    query_id = "genQ" + str(count)
                    self.queries[query_id] = query
                    self.qrels[query_id] = {corpus_id: 1}

        # Saving finally all the questions
        logger.info("Saving {} Generated Queries...".format(len(self.queries)))
        self.save(output_dir, self.queries, self.qrels, prefix)

    def generate_multi_process(
        self,
        corpus: Dict[str, Dict[str, str]],
        pool: Dict[str, object],
        output_dir: str,
        top_p: int = 0.95,
        top_k: int = 25,
        max_length: int = 64,
        ques_per_passage: int = 1,
        prefix: str = "gen",
        batch_size: int = 32,
        chunk_size: int = None
    ):

        logger.info(
            "Starting to Generate {} Questions Per Passage using top-p (nucleus) sampling...".format(ques_per_passage)
        )
        logger.info("Params: top_p = {}".format(top_p))
        logger.info("Params: top_k = {}".format(top_k))
        logger.info("Params: max_length = {}".format(max_length))
        logger.info("Params: ques_per_passage = {}".format(ques_per_passage))
        logger.info("Params: batch size = {}".format(batch_size))

        count = 0
        corpus_ids = list(corpus.keys())
        corpus = [corpus[doc_id] for doc_id in corpus_ids]

        queries = self.model.generate_multi_process(
            corpus=corpus,
            pool=pool,
            ques_per_passage=ques_per_passage,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )

        assert len(queries) == len(corpus) * ques_per_passage

        for idx in range(len(corpus)):
            corpus_id = corpus_ids[idx]
            start_id = idx * ques_per_passage
            end_id = start_id + ques_per_passage
            query_set = set([q.strip() for q in queries[start_id:end_id]])

            for query in query_set:
                count += 1
                query_id = "genQ" + str(count)
                self.queries[query_id] = query
                self.qrels[query_id] = {corpus_id: 1}

        # Saving finally all the questions
        logger.info("Saving {} Generated Queries...".format(len(self.queries)))
        self.save(output_dir, self.queries, self.qrels, prefix)


def qgen(
    data_path,
    output_dir,
    generator_name_or_path='BeIR/query-gen-msmarco-t5-base-v1',
    ques_per_passage=3,
    bsz=32,
    qgen_prefix='qgen'
):
    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    corpus = GenericDataLoader(data_path).load_corpus()

    #### question-generation model loading
    generator = QueryGenerator(model=QGenModel(generator_name_or_path))

    #### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
    #### https://huggingface.co/blog/how-to-generate
    #### Prefix is required to seperate out synthetic queries and qrels from original
    prefix = qgen_prefix

    #### Generating 3 questions per passage.
    #### Reminder the higher value might produce lots of duplicates
    #### Generate queries per passage from docs in corpus and save them in data_path
    generator.generate(
        corpus,
        output_dir=output_dir,
        ques_per_passage=ques_per_passage,
        prefix=prefix,
        batch_size=bsz,
        beam_search=True
    )
    if not os.path.exists(os.path.join(output_dir, 'corpus.jsonl')):
        os.system(f'cp {data_path}/corpus.jsonl {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    qgen(args.data_path, args.output_dir)
