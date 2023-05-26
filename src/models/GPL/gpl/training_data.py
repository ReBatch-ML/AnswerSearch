'''
Script that generates synthetic training data from preprocessed paragraphs
Most important arguments found in GPL/gpl/gpl_args.yml
'''

from typing import List, Union
from kfp.v2.dsl import Dataset, Input


def generate_gpl_data(
    path_to_generated_data: str,
    output_dir: str,
    corpus_path_on_datastore: Input[Dataset],
    mnrl_output_dir: str = None,
    mnrl_evaluation_output: str = None,
    do_evaluation: str = False,
    evaluation_data: str = None,
    evaluation_output: str = 'output',
    qgen_prefix: str = 'qgen',
    base_ckpt: str = 'distilbert-base-uncased',
    generator: str = 'BeIR/query-gen-msmarco-t5-base-v1',
    cross_encoder: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    batch_size_gpl: int = 32,
    batch_size_generation: int = 32,
    pooling: str = None,
    max_seq_length: int = 350,
    new_size: int = None,
    queries_per_passage: int = 3,
    gpl_steps: int = 140000,
    use_amp: bool = False,
    retrievers: List[str] = ['msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3'],
    retriever_score_functions: List[str] = ['cos_sim', 'cos_sim'],
    negatives_per_query: int = 50,
    eval_split: str = 'test',
    use_train_qrels: bool = False,
    gpl_score_function: str = 'dot',
    rescale_range: List[float] = None,
):
    """_summary_

    Args:
        path_to_generated_data (str): _description_
        output_dir (str): _description_
        corpus_path_on_datastore (Input[Dataset]): _description_
        mnrl_output_dir (str, optional): _description_. Defaults to None.
        mnrl_evaluation_output (str, optional): _description_. Defaults to None.
        do_evaluation (str, optional): _description_. Defaults to False.
        evaluation_data (str, optional): _description_. Defaults to None.
        evaluation_output (str, optional): _description_. Defaults to 'output'.
        qgen_prefix (str, optional): _description_. Defaults to 'qgen'.
        base_ckpt (str, optional): _description_. Defaults to 'distilbert-base-uncased'.
        generator (str, optional): _description_. Defaults to 'BeIR/query-gen-msmarco-t5-base-v1'.
        cross_encoder (str, optional): _description_. Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        batch_size_gpl (int, optional): _description_. Defaults to 32.
        batch_size_generation (int, optional): _description_. Defaults to 32.
        pooling (str, optional): _description_. Defaults to None.
        max_seq_length (int, optional): _description_. Defaults to 350.
        new_size (int, optional): _description_. Defaults to None.
        queries_per_passage (int, optional): _description_. Defaults to 3.
        gpl_steps (int, optional): _description_. Defaults to 140000.
        use_amp (bool, optional): _description_. Defaults to False.
        retrievers (List[str], optional): _description_. Defaults to ['msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3'].
        retriever_score_functions (List[str], optional): _description_. Defaults to ['cos_sim', 'cos_sim'].
        negatives_per_query (int, optional): _description_. Defaults to 50.
        eval_split (str, optional): _description_. Defaults to 'test'.
        use_train_qrels (bool, optional): _description_. Defaults to False.
        gpl_score_function (str, optional): _description_. Defaults to 'dot'.
        rescale_range (List[float], optional): _description_. Defaults to None.

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    #### IMPORTS ###
    import shutil
    from beir.datasets.data_loader import GenericDataLoader
    from gpl.toolkit import (
        qgen, NegativeMiner, PseudoLabeler, resize, set_logger_format, save_queries, save_qrels, extract_queries_split,
        rescale_gpl_training_data
    )
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import numpy as np
    from numpy.linalg import norm
    import torch
    from torch.utils.data import DataLoader
    import os
    import logging
    import argparse
    from typing import List, Union
    import math
    import glob
    import shutil
    import pandas as pd
    import jsonlines
    import random
    from heapq import nlargest

    from google.cloud import storage

    set_logger_format()
    logger = logging.getLogger(
        'gpl.train'
    )  # Here we do not use __name__ to have unified logger name, no matter whether we are using `python -m` or `import gpl; gpl.train`
    GCS_CLIENT = storage.Client()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)

    #### CUSTOM FUNCTIONS ###
    def prepend_title(line):
        """_summary_

        Args:
            line (_type_): _description_

        Returns:
            _type_: _description_
        """
        title = line['title']
        line['text'] = title + ': ' + line['text']
        # if prepend, title must be empty because Qgen looks here for making question
        line['title'] = ''
        return line

    def sample_from_iterable(iterable, samplesize):
        """_summary_

        Args:
            iterable (_type_): _description_
            samplesize (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (x for _, x in nlargest(samplesize, ((random.random(), x) for x in iterable)))

    # there is simply too much data to use everything for training purpose and not all paragraphs are of highest quality
    # keep num_passages of paragraphs and possibly prepend title
    def copy_corpus_and_filter(source_path, dest_path, num_passages=500000, prepend=False):
        """_summary_

        Args:
            source_path (_type_): _description_
            dest_path (_type_): _description_
            num_passages (int, optional): _description_. Defaults to 500000.
            prepend (bool, optional): _description_. Defaults to False.
        """
        minimum_paragraph_length = 500
        count_written = 0
        with jsonlines.open(dest_path, mode='w') as writer:
            with jsonlines.open(source_path) as reader:
                paragraphs = sample_from_iterable(reader, int(num_passages * 3))
                for paragraph in paragraphs:
                    if count_written > num_passages:
                        break
                    l = len(paragraph['text'])
                    if l > minimum_paragraph_length:
                        count_written += 1
                        if prepend:
                            writer.write(prepend_title(paragraph))
                        else:
                            writer.write(paragraph)
        print(
            f'Corpus copied to {dest_path}. {count_written} paragraphs longer than {minimum_paragraph_length} chars were kept. Prepend title = {prepend}'
        )

    # Keep only selection of all training/test data
    # choose either bi-encoder or cross-encoder to keep only queries that are 'good' enough (threshold ~5 for cross and ~0.7 for bi-enc with cos_sim)
    def postprocess_queries(path_to_data, threshold=5, cross=True):
        """_summary_

        Args:
            path_to_data (_type_): _description_
            threshold (int, optional): _description_. Defaults to 5.
            cross (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        qgen_df = pd.read_json(path_or_buf=f'{path_to_data}/qgen-queries.jsonl', lines=True)
        qgen_qrel_df = pd.read_csv(f'{path_to_data}/qgen-qrels/train.tsv', sep='\t')
        corpus_df = pd.read_json(path_or_buf=f'{path_to_data}/corpus.jsonl', lines=True)

        def cos_sim(A, B):
            """_summary_

            Args:
                A (_type_): _description_
                B (_type_): _description_

            Returns:
                _type_: _description_
            """
            cosine = np.einsum('ij, ij->i', A, B) / (norm(A, axis=1) * norm(B, axis=1))
            return cosine

        corpus_lookup = pd.Series(corpus_df.text.values, index=corpus_df._id).to_dict()
        qgen_pos_lookup = pd.Series(qgen_qrel_df['corpus-id'].values, index=qgen_qrel_df['query-id'].values).to_dict()
        q_txt = qgen_df.text.values
        pos_txt = qgen_df['_id'].map(lambda x: corpus_lookup[qgen_pos_lookup[x]])

        if cross:
            print('Using cross-encoder to filter queries ...')
            encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            cross_input = list(zip(q_txt, pos_txt))
            scores = encoder.predict(cross_input, show_progress_bar=True)

        else:
            print('Using bi-encoder to filter queries ...')
            encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
            q_embs = encoder.encode(q_txt, batch_size=16, show_progress_bar=True)
            pos_embs = encoder.encode(pos_txt.values, batch_size=16, show_progress_bar=True)
            scores = cos_sim(pos_embs, q_embs)

        indxs_to_drop = []
        for i, score in enumerate(scores):
            if score < threshold:
                indxs_to_drop.append(i)

        qgen_qrel_df.drop(indxs_to_drop, axis=0, inplace=True)
        qgen_filterd_df = qgen_df[qgen_df['_id'].isin(qgen_qrel_df['query-id'].values)]

        qgen_filterd_df.to_json(f'{path_to_data}/qgen-queries.jsonl', orient='records', lines=True)
        qgen_qrel_df.to_csv(f'{path_to_data}/qgen-qrels/train.tsv', sep='\t', index=False)

        print(len(qgen_filterd_df.index), 'queries kept out of all', len(qgen_df.index), 'Threshold at', threshold)
        return

    # split all generated data into train and test
    # start by splitting queries and based on this split the hard-negatives, the effective training lines and qrels
    def upload_train_test_split(train_fraction, filename, path_to_data, bucket):
        """_summary_

        Args:
            train_fraction (_type_): _description_
            filename (_type_): _description_
            path_to_data (_type_): _description_
            bucket (_type_): _description_
        """
        # read generated files as dataframe to make train-test-split, delete originals to save diskspace
        queries_df = pd.read_json(path_or_buf=f'{path_to_data}/qgen-queries.jsonl', lines=True)
        hn_df = pd.read_json(path_or_buf=f'{path_to_data}/hard-negatives.jsonl', lines=True)
        gpl_train_df = pd.read_csv(f'{path_to_data}/gpl-training-data.tsv', header=None, sep='\t')
        qgen_qrel_df = pd.read_csv(f'{path_to_data}/qgen-qrels/train.tsv', sep='\t')
        #corpus_df = pd.read_json(path_or_buf=f'{path_to_data}/corpus.jsonl', lines=True)

        # split fraction of generated queries, use these queries to split other dataframes
        queries_train = queries_df.sample(frac=train_fraction, random_state=200)  #random state is a seed value
        queries_test = queries_df.drop(queries_train.index)
        q = queries_train['_id'].values

        hn_train = hn_df[hn_df['qid'].isin(q)]
        hn_test = hn_df.drop(hn_train.index)

        gpl_train = gpl_train_df[gpl_train_df[0].isin(q)]
        qgen_qrel_train = qgen_qrel_df[qgen_qrel_df['query-id'].isin(q)]
        qgen_qrel_test = qgen_qrel_df.drop(qgen_qrel_train.index)

        # make test and train dirs
        train_dir = path_to_data.rsplit('/', 1)[0] + '/train'
        test_dir = path_to_data.rsplit('/', 1)[0] + '/test'
        os.makedirs(train_dir + '/qgen-qrels', exist_ok=True)
        os.makedirs(test_dir + '/qrels', exist_ok=True)

        # save files to relevant dirs
        queries_test.to_json(f'{test_dir}/queries.jsonl', orient='records', lines=True)
        queries_train.to_json(f'{train_dir}/qgen-queries.jsonl', orient='records', lines=True)

        hn_test.to_json(f'{test_dir}/hard-negatives.jsonl', orient='records', lines=True)
        hn_train.to_json(f'{train_dir}/hard-negatives.jsonl', orient='records', lines=True)

        gpl_train.to_csv(f'{train_dir}/gpl-training-data.tsv', header=False, sep='\t', index=False)
        qgen_qrel_test.to_csv(f'{test_dir}/qrels/test.tsv', sep='\t', index=False)
        qgen_qrel_train.to_csv(f'{train_dir}/qgen-qrels/train.tsv', sep='\t', index=False)
        shutil.move(f"{path_to_data}/corpus.jsonl", f"{train_dir}/corpus.jsonl")
        shutil.copy(f"{train_dir}/corpus.jsonl", f"{test_dir}/corpus.jsonl")

    #### Assertions ####
    assert pooling in [None, 'mean', 'cls', 'max']
    if do_evaluation:
        assert evaluation_data is not None
        assert evaluation_output is not None
        try:
            GenericDataLoader(evaluation_data)
        except Exception as e:
            logger.error('Cannot load evaluation data for evaluation usage.')
            raise e
    if new_size is not None and new_size != -1:
        assert new_size * queries_per_passage >= batch_size_gpl

    # rename jsonl file to corpus.jsonl as this script looks for it explicitly.
    corpus_path = corpus_path_on_datastore.path
    filepath = glob.glob('{}'.format(corpus_path))[0]
    filename = filepath.split('.')[0].split('/')[-1]
    print('corpus filepath', filepath)

    os.makedirs(path_to_generated_data, exist_ok=True)
    files_in_gen = glob.glob('{}/**'.format(path_to_generated_data), recursive=True)
    print('Files in generated data directory:', files_in_gen)

    #shutil.move(filepath.replace(filename, 'corpus'), os.path.join(path_to_generated_data, 'corpus.jsonl') )
    if 'corpus.jsonl' not in os.listdir(path_to_generated_data):
        print('corpus not found in directory, copying from corpus input path')
        copy_corpus_and_filter(source_path=filepath, dest_path=path_to_generated_data + '/corpus.jsonl', prepend=True)

    #### Make sure there is a `corpus.jsonl` file. It should be under either `path_to_generated_data` or `evaluation_data`` ####
    #### Also resize the corpus for efficient training if required  ####

    if 'corpus.jsonl' not in os.listdir(path_to_generated_data):
        print(
            f'Corpus does not exist in {path_to_generated_data}. Now clone the one from the evaluation path {evaluation_data}'
        )
        assert 'corpus.jsonl' in os.listdir(
            evaluation_data
        ), f'No corpus found in evaluation path {evaluation_data}! It should be in the BeIR format. For more details, please refer to https://github.com/UKPLab/beir#beers-available-datasets.'
        if new_size is not None:
            if new_size == -1:
                new_size = math.ceil(250e3 / 3)  # Here use ceil to make the QPP == 3 if the corpus is large enough
                print(f'Automatically set `new_size` to {new_size}')
            resize(evaluation_data, path_to_generated_data, new_size, use_train_qrels)
        else:
            corpus_path = os.path.join(evaluation_data, 'corpus.jsonl')
            os.system(f'cp {corpus_path} {path_to_generated_data}')

    #### Adjust the QQP automatically, if needed ####
    if queries_per_passage == -1:
        assert 'corpus.jsonl' in os.listdir(path_to_generated_data), 'At least corpus should exist!'
        corpus = GenericDataLoader(path_to_generated_data).load_corpus()
        if len(corpus) * 3 < 250e3:
            queries_per_passage = math.ceil(
                250e3 / len(corpus)
            )  # Here use ceil to guarantee the QPP will not be too small
        else:
            queries_per_passage = 3
        print(f'Automatically set `queries_per_passage` to {queries_per_passage}')

    #### Synthetic query generation ####
    #### This will be skipped if there is an existing `gen-queries.jsonl`file under `path_to_generated_data` ####
    print('Start query generation')
    if use_train_qrels == True:
        if qgen_prefix is not None:
            logger.warning(
                'Found `qgen_prefix` is not None. By setting `use_train_qrels == True`, the `qgen_prefix` will not be used'
            )

        if 'qrels' in os.listdir(path_to_generated_data) and 'queries.jsonl' in os.listdir(path_to_generated_data):
            print('Loading from existing labeled data')
            corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data).load(split="train")
        else:
            assert evaluation_data is not None, "To use this feature `use_train_qrels == True`, please specify the `evaluation_data`, which should contain the labeled queries and qrels"
            print('Loading qrels and queries from labeled data under the path of `evaluation_data`')
            assert 'qrels' in os.listdir(evaluation_data) and 'queries.jsonl' in os.listdir(evaluation_data)
            assert 'train.tsv' in os.listdir(os.path.join(evaluation_data, 'qrels'))
            corpus, all_queries, train_qrels = GenericDataLoader(evaluation_data).load(
                split='train'
            )  # TODO: Change the variable name `gen_queries`
            train_queries = extract_queries_split(all_queries, train_qrels)
            save_queries(
                train_queries, path_to_generated_data
            )  # Copy the training data into the `path_to_generated_data` folder,
            save_qrels(
                train_qrels, path_to_generated_data, split='train'
            )  # then the negative miner can load it and run mining thereon
            gen_queries = train_queries  # This variable will be passed into the PseudoLabeler
    elif f'{qgen_prefix}-qrels' in os.listdir(path_to_generated_data) and f'{qgen_prefix}-queries.jsonl' in os.listdir(
        path_to_generated_data
    ):
        print('Loading from existing generated data')
        corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data,
                                                           prefix=qgen_prefix).load(split="train")
    else:
        print('No generated queries found. Now generating it')
        assert 'corpus.jsonl' in os.listdir(path_to_generated_data), 'At least corpus should exist!'
        qgen(
            path_to_generated_data,
            path_to_generated_data,
            generator_name_or_path=generator,
            ques_per_passage=queries_per_passage,
            bsz=batch_size_generation,
            qgen_prefix=qgen_prefix
        )
        print('QGEN completed, now filtering out weak queries')
        postprocess_queries(path_to_generated_data, threshold=5, cross=True)
        corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data,
                                                           prefix=qgen_prefix).load(split="train")

    #### Hard-negative mining ####
    #### This will be skipped if there is an existing `hard-negatives.jsonl` file under `path_to_generated_data` ####
    print('start hard-negative mining')
    if 'hard-negatives.jsonl' in os.listdir(path_to_generated_data):
        print('Using exisiting hard-negative data')
    else:
        print('No hard-negative data found. Now mining it')
        miner = NegativeMiner(
            path_to_generated_data,
            qgen_prefix,
            retrievers=retrievers,
            retriever_score_functions=retriever_score_functions,
            nneg=negatives_per_query,
            use_train_qrels=use_train_qrels
        )
        miner.run()

    #### Pseudo labeling ####
    #### This will be skipped if there is an existing `gpl-training-data.tsv` file under `path_to_generated_data` ####
    print('start pseudo labeling')
    gpl_training_data_fname = 'gpl-training-data.tsv'
    if gpl_training_data_fname in os.listdir(path_to_generated_data):
        print('Using existing GPL-training data')
    else:
        print('No GPL-training data found. Now generating it via pseudo labeling')
        try:
            pseudo_labeler = PseudoLabeler(
                path_to_generated_data, gen_queries, corpus, gpl_steps, batch_size_gpl, cross_encoder, max_seq_length
            )
            pseudo_labeler.run()
        except Exception as e:
            print(type(e))
            print(e.args)
            print(e)

    # Do rescaling if needed:
    if rescale_range is not None and len(rescale_range) == 2:
        if gpl_score_function != 'cos_sim':
            logger.warning(f'Doing rescaling while gpl_score_function = {gpl_score_function}')

        new_min, new_max = rescale_range
        print(f'Doing rescaling with new range [{new_min}, {new_max}]')
        gpl_training_data_fname = rescale_gpl_training_data(
            path_to_generated_data, new_min, new_max
        )  # This will rescale the margins and generate a new file
    else:
        # if len(rescale_range) != 2:
        #     logger.warning(f'len(rescale_range) should be 2')
        if gpl_score_function == 'cos_sim':
            logger.warning(f'Not do rescaling while gpl_score_function = {gpl_score_function}')

    # Upload all generated files to cloud storage
    print('uploading files to cloud storage')
    upload_train_test_split(
        train_fraction=0.98, filename=filename, path_to_data=path_to_generated_data, bucket='semantic_search_blobstore'
    )


# DEPRECATED
# Used to work with generate_gpl_data pipeline on Azure
# now generate_gpl_data definition is used directly in gcp pipeline
if __name__ == '__main__':
    run = Run.get_context()
    ws = run.experiment.workspace

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_generated_data',
        required=True,
        help=
        'Path for/to the generated data. GPL will first check this path for a `corpus.jsonl` file for the (sole) data input of the whole pipeline. If an empty folder is indicated, query generation and hard-negative mining will be run automatically; one can also use a BeIR-QGen format data folder to start and skip the query generation.'
    )
    parser.add_argument('--output_dir', required=True, help='Output path for the GPL model.')
    parser.add_argument(
        '--do_evaluation', action='store_true', default=False, help='Wether to do the evaluation (after training)'
    )
    parser.add_argument(
        '--evaluation_data',
        type=str,
        help=
        'Path to the BeIR-format dataset. This is the next folder GPL goes to for the target corpus if there is no `corpus.jsonl` under `path_to_generated_data`'
    )
    parser.add_argument('--evaluation_output', default='output', help='Path for the evaluation output.')
    parser.add_argument(
        '--qgen_prefix',
        default='qgen',
        help=
        'This prefix will appear as part of the (folder/file) names for query-generation results: For example, we will have "qgen-qrels/" and "qgen-queries.jsonl" by default.'
    )
    parser.add_argument(
        '--base_ckpt',
        default='distilbert-base-uncased',
        help='Initialization checkpoint in HF or SBERT format. Meaning-pooling will be used.'
    )
    parser.add_argument('--generator', default='BeIR/query-gen-msmarco-t5-base-v1')
    parser.add_argument('--cross_encoder', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--batch_size_gpl', type=int, default=32)
    parser.add_argument(
        '--batch_size_generation', type=int, default=10, help='Batch size in the query generation step.'
    )
    parser.add_argument(
        '--pooling',
        type=str,
        default=None,
        choices=['cls', 'mean', 'max'],
        help=
        "Specifying pooling method for dense retriever if in Huggingface-format. By default (None), it uses mean pooling. If in SBERT-format, there would be the indicated pooling method in its configure file and thus this argument will be ignored. "
    )
    parser.add_argument('--max_seq_length', type=int, default=350)
    parser.add_argument(
        '--new_size',
        type=int,
        default=None,
        help=
        'Resize the corpus to `new_size` (|corpus|) if needed. When set to None (by default), the |corpus| will be the full size. When set to -1, the |corpus| will be set automatically: If QPP * |corpus| <= 250K, |corpus| will be the full size; else QPP will be set 3 and |corpus| will be set to 250K / 3'
    )
    parser.add_argument(
        '--queries_per_passage',
        type=int,
        default=-1,
        help=
        'Number of Queries Per Passage (QPP) in the query generation step. When set to -1 (by default), the QPP will be chosen automatically: If QPP * |corpus| <= 250K, then QPP will be set to 250K / |corpus|; else QPP will be set 3 and |corpus| will be set to 250K / 3'
    )
    parser.add_argument('--gpl_steps', type=int, default=140000, help='Training steps for GPL.')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Whether to use half precision')
    parser.add_argument(
        '--retrievers',
        nargs='+',
        default=['msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3'],
        help=
        'Indicate retriever names for mining negatives. They could be one or many BM25 ("bm25") or dense retrievers (in SBERT format).'
    )
    parser.add_argument(
        '--retriever_score_functions',
        nargs='+',
        default=['cos_sim', 'cos_sim'],
        choices=['dot', 'cos_sim', 'none'],
        help='Score functions of the corresponding retrievers for negative mining. Please set it to "none" for BM25.'
    )
    parser.add_argument('--gpl_score_function', choices=['dot', 'cos_sim'], default='dot')
    parser.add_argument(
        '--rescale_range',
        nargs='+',
        type=float,
        default=None,
        help=
        'Rescale the pseudo labels (i.e. score margins) to a certain range. For example, we can set this to "-2 2", which represents the margin range based on cosine-similarity. By default, it will not do rescaling.'
    )
    parser.add_argument(
        '--negatives_per_query', type=int, default=50, help="Mine how many negatives per query per retriever"
    )
    parser.add_argument('--mnrl_output_dir', default=None)
    parser.add_argument('--mnrl_evaluation_output', default=None)
    parser.add_argument(
        '--eval_split', type=str, default='test', choices=['train', 'test', 'dev'], help='Which split to evaluate on'
    )
    parser.add_argument('--use_train_qrels', action='store_true', default=False)
    parser.add_argument('--corpus_path_on_datastore', type=str)
    args = parser.parse_args()

    # rename jsonl file to corpus.jsonl as this script looks for it explicitly.
    print(args.corpus_path_on_datastore)
    filepath = glob.glob('{}'.format(args.corpus_path_on_datastore), recursive=True)[0]
    filename = filepath.split('.')[0].split('/')[-1]
    os.rename(filepath, filepath.replace(filename, 'corpus'))

    # start to generate queries, hard-negatives and pseudolabels
    generate_gpl_data(**vars(args))

    # upload results to datastore, split data in train and testset
    upload_train_test_split(train_fraction=0.9, filename=filename, path_to_data=args.path_to_generated_data)
