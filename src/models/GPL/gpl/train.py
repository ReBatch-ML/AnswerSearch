'''
Script that starts GPL training based on synthetic training data generated in GPL/gpl/training_data.py
Most important arguments found in GPL/gpl/gpl_args.yml
'''
from typing import List, Union
from kfp.v2.dsl import Dataset, Input


def train(
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
        e: _description_

    Returns:
        _type_: _description_
    """

    print('getting imports')
    #### IMPORTS ######
    import shutil
    from beir.datasets.data_loader import GenericDataLoader
    from gpl.toolkit import (
        qgen, NegativeMiner, MarginDistillationLoss, GenerativePseudoLabelingDataset, PseudoLabeler, resize, load_sbert,
        set_logger_format, mnrl, save_queries, save_qrels, extract_queries_split, rescale_gpl_training_data
    )
    from sentence_transformers import SentenceTransformer, evaluation

    from torch.utils.data import DataLoader
    import numpy as np
    from beir.retrieval import models
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    from beir.retrieval.evaluation import EvaluateRetrieval

    import os
    import logging
    import argparse
    from typing import List, Union
    import math
    from glob import glob
    import shutil
    import pandas as pd
    from datetime import datetime
    import json
    import torch

    from google.cloud import secretmanager
    from google.cloud import storage

    # from azureml.core import Run, Dataset, Datastore
    # from azureml.data.datapath import DataPath
    # from azureml.data import FileDataset
    from azureml.core import Workspace, Dataset
    from azureml.core import Run, Model
    from azureml.core.authentication import ServicePrincipalAuthentication

    ############################ FUNCTIONS ####################################

    # compute recall, precision, mAP, and ndcg
    def evaluate(
        data_path: str,
        output_dir: str,
        model_name_or_path: str,
        max_seq_length: int = 512,
        score_function: str = 'dot',
        pooling: str = None,
        sep: str = ' ',
        k_values: List[int] = [10, 30, 50, 100],
        split: str = 'test'
    ):
        """_summary_

        Args:
            data_path (str): _description_
            output_dir (str): _description_
            model_name_or_path (str): _description_
            max_seq_length (int, optional): _description_. Defaults to 512.
            score_function (str, optional): _description_. Defaults to 'dot'.
            pooling (str, optional): _description_. Defaults to None.
            sep (str, optional): _description_. Defaults to ' '.
            k_values (List[int], optional): _description_. Defaults to [10, 30, 50, 100].
            split (str, optional): _description_. Defaults to 'test'.

        Returns:
            _type_: _description_
        """
        model: SentenceTransformer = load_sbert(model_name_or_path, pooling, max_seq_length)

        data_paths = []
        data_paths.append(data_path)

        ndcgs = []
        _maps = []
        recalls = []
        precisions = []
        for data_path in data_paths:
            corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

            sbert = models.SentenceBERT(sep=sep)
            sbert.q_model = model
            sbert.doc_model = model

            model_dres = DRES(sbert, batch_size=16, corpus_chunk_size=18000)
            assert score_function in ['dot', 'cos_sim']
            retriever = EvaluateRetrieval(
                model_dres, score_function=score_function, k_values=k_values
            )  # or "dot" for dot-product
            results = retriever.retrieve(corpus, queries)

            #### Evaluate your retrieval using NDCG@k, MAP@K ...
            ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
            ndcgs.append(ndcg)
            _maps.append(_map)
            recalls.append(recall)
            precisions.append(precision)

        ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
        _map = {k: np.mean([score[k] for score in _maps]) for k in _map}
        recall = {k: np.mean([score[k] for score in recalls]) for k in recall}
        precision = {k: np.mean([score[k] for score in precisions]) for k in precision}

        metrics = dict(**ndcg, **_map, **recall, **precision)
        return metrics

    # Model Callback to save on Azure
    def save_callback(score, epoch, steps):
        """_summary_

        Args:
            score (_type_): _description_
            epoch (_type_): _description_
            steps (_type_): _description_

        Raises:
            e: _description_
        """
        print(checkpoint_dir)
        subdirs = glob(checkpoint_dir + '/*')
        print('checkpoints:', subdirs)
        if len(subdirs) > 0:
            ckpts = [int(ckpt) for ckpt in list(map(lambda x: x.rsplit('/', 1)[-1], subdirs)) if ckpt.isdigit()]
            latest_ckpt = str(max(ckpts))

            model_out_dir = checkpoint_dir + '/' + latest_ckpt
            properties = {'total_gpl_steps': gpl_steps, 'path_to_data': path_to_generated_data}

            print('Start Evaluation')
            try:
                metrics = evaluate(
                    data_path=evaluation_data,
                    output_dir=output_dir,
                    model_name_or_path=model_out_dir,
                )
            except Exception as e:
                print(e)
                print(type(e))
                raise e
            print('metrics:', metrics)

            Model.register(
                workspace=ws, model_path=model_out_dir, model_name=model_name, properties=dict(**metrics, **properties)
            )

            # delete the checkpoint folders up till now
            for folder in subdirs:
                shutil.rmtree(folder)

    ####################################################################################################################################################################################

    DATE = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = base_ckpt + '_TRAINED_' + DATE
    model_name = model_name.replace('/', '_')
    print("model_name:", model_name)

    set_logger_format()
    logger = logging.getLogger(
        'gpl.train'
    )  # Here we do not use __name__ to have unified logger name, no matter whether we are using `python -m` or `import gpl; gpl.train`
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #### Load AzureML workspace
    # load azure service principal
    secret_client = secretmanager.SecretManagerServiceClient()
    response = secret_client.access_secret_version(
        request={"name": 'projects/rebatch-sandbox-329013/secrets/azure_service_principal_ss/versions/1'}
    )
    sp = json.loads(response.payload.data.decode("UTF-8"))
    sp_auth = ServicePrincipalAuthentication(
        tenant_id=sp["tenant"], service_principal_id=sp["appId"], service_principal_password=sp["password"]
    )
    print('service principal loaded')
    # get workspace
    subscription_id = "9da3a5d6-6bf3-4b2c-8219-88caf39f718d"
    resource_group = "Semantic_Search"
    workspace_name = "SemanticSearch_TRAIN"
    ws = Workspace(subscription_id, resource_group, workspace_name, auth=sp_auth)
    print(ws, ' obtained from Azure')

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

    #### Make sure there is a `corpus.jsonl` file. It should be under either `path_to_generated_data` or `evaluation_data`` ####
    #### Also resize the corpus for efficient training if required  ####
    print('path to data', path_to_generated_data)
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

    if f'{qgen_prefix}-qrels' in os.listdir(path_to_generated_data
                                            ) and f'{qgen_prefix}-queries.jsonl' in os.listdir(path_to_generated_data):
        print('Loading from existing generated data')
        corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data,
                                                           prefix=qgen_prefix).load(split="train")
    else:
        print('No generated queries found!')

    if 'hard-negatives.jsonl' in os.listdir(path_to_generated_data):
        print('Using exisiting hard-negative data')
    else:
        print('No hard-negative data found!')

    #### Pseudo labeling ####
    #### This will be skipped if there is an existing `gpl-training-data.tsv` file under `path_to_generated_data` ####
    gpl_training_data_fname = 'gpl-training-data.tsv'
    if gpl_training_data_fname in os.listdir(path_to_generated_data):
        print('Using existing GPL-training data')
    else:
        print('No GPL-training data found!')

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

    ### Train the model with MarginMSE loss ###
    #### This will be skipped if the checkpoint at the indicated training steps can be found ####
    print('Start Training')
    ckpt_dir = os.path.join(output_dir, str(gpl_steps))
    if not os.path.exists(ckpt_dir) or (os.path.exists(ckpt_dir) and not os.listdir(ckpt_dir)):
        print('Now doing training on the generated data with the MarginMSE loss')
        #### It can load checkpoints in both SBERT-format (recommended) and Huggingface-format
        model: SentenceTransformer = load_sbert(base_ckpt, pooling, max_seq_length)

        fpath_gpl_data = os.path.join(path_to_generated_data, gpl_training_data_fname)
        print(f'Load GPL training data from {fpath_gpl_data}')
        train_dataset = GenerativePseudoLabelingDataset(fpath_gpl_data, gen_queries, corpus)
        train_dataloader = DataLoader(
            train_dataset, shuffle=False, batch_size=batch_size_gpl, drop_last=True
        )  # Here shuffle=False, since (or assuming) we have done it in the pseudo labeling
        train_loss = MarginDistillationLoss(model=model, similarity_fct=gpl_score_function)
        evaluator = evaluation.TripletEvaluator(
            anchors=['mock_query'], positives=['mock_positive'], negatives=['mock_negative']
        )  # without evaluator callback won't work

        checkpoint_dir = output_dir + 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        eval_save_steps = 15000
        # assert gpl_steps > 1000

        model.fit(
            [
                (train_dataloader, train_loss),
            ],
            epochs=1,
            steps_per_epoch=gpl_steps,
            evaluator=evaluator,
            evaluation_steps=eval_save_steps,
            warmup_steps=1000,
            checkpoint_save_steps=eval_save_steps - 1,
            checkpoint_save_total_limit=10000,
            output_path=output_dir,
            checkpoint_path=checkpoint_dir,
            use_amp=use_amp,
            callback=save_callback,
            save_best_model=False,
        )

    else:
        print('Trained GPL model found. Now skip training')

    # ### Evaluate the model if required ###
    # if do_evaluation:
    #     print('Doing evaluation for GPL')
    #     evaluate(
    #         evaluation_data,
    #         evaluation_output,
    #         ckpt_dir,
    #         max_seq_length,
    #         score_function=gpl_score_function,
    #         pooling=pooling,
    #         split=eval_split
    #     )


# DEPRECATED
# Used to work with run_training pipeline on Azure
# now train definition is used directly in gcp pipeline
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
    args = parser.parse_args()

    print(os.listdir())
    print(args.path_to_generated_data)
    print(os.listdir(args.path_to_generated_data))

    # start to generate queries, hard-negatives and pseudolabels
    train(**vars(args))

    # upload results to datastore, split data in train and testset
    # upload_train_test_split(train_fraction=0.9, filename=filename, path_to_data=args.path_to_generated_data)
