'''
Evaluate model on synthetically generated queries
Takes as input:
- bi-encoder (trained or untrained)
- generated queries: path to GPL generated data, will select test data from this folder
- dataset hf: huggingface dataset containing info and embeddings on the entire train+test set
'''
from kfp.v2.dsl import Dataset, Input


def evaluate_corpus(bi_encoder: str, generated_queries: Input[Dataset], dataset_hf: Input[Dataset]):
    """_summary_

    Args:
        bi_encoder (str): _description_
        generated_queries (Input[Dataset]): _description_
        dataset_hf (Input[Dataset]): _description_

    Returns:
        _type_: _description_
    """

    import numpy as np
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    import pandas as pd
    import glob
    from datasets import load_from_disk, Dataset
    import datasets
    import time
    import os
    import json
    from google.cloud import secretmanager
    from google.cloud import storage
    import faiss
    import torch

    from azureml.core import Workspace, Dataset
    from azureml.core import Run, Model
    from azureml.core.authentication import ServicePrincipalAuthentication

    #### Load AzureML workspace
    # load azure service principal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    secret_client = secretmanager.SecretManagerServiceClient()
    response = secret_client.access_secret_version(
        request={"name": 'projects/rebatch-sandbox-329013/secrets/azure_service_principal_ss/versions/1'}
    )
    sp = json.loads(response.payload.data.decode("UTF-8"))
    print('loading service principal')
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

    class QueryEvaluator():
        """_summary_
        """

        def __init__(self, data_dir, bi_encoder_name):
            test_dir = 'test'
            self.hn_df = pd.read_json(path_or_buf=os.path.join(data_dir, test_dir, 'hard-negatives.jsonl'), lines=True)
            self.qgen_df = pd.read_json(path_or_buf=os.path.join(data_dir, test_dir, 'queries.jsonl'), lines=True)
            self.corpus_df = pd.read_json(path_or_buf=os.path.join(data_dir, test_dir, 'corpus.jsonl'), lines=True)

            try:
                bi_encoder_name, version = bi_encoder_name.split('_version_')
                model = Model(workspace=ws, name=bi_encoder_name, version=int(version)).download(exist_ok=True)
                bi_encoder = SentenceTransformer(model)
                print(f'Model {bi_encoder_name} version {version} loaded from ws')
            except Exception as e:
                print(e)
                print(f'Loading pretrained {bi_encoder_name} from Sentence-Transformers')
                bi_encoder = SentenceTransformer(bi_encoder_name)

            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            bi_encoder.max_seq_length = 512  #Truncate long passages to 512 tokens
            bi_encoder.tokenizer.truncate_sequences = False
            self.tokenizer = bi_encoder.tokenizer
            self.bi_encoder = bi_encoder

        def search_faiss_bi_encoder(self, ds, question_embedding):
            """_summary_

            Args:
                ds (_type_): _description_
                question_embedding (_type_): _description_

            Returns:
                _type_: _description_
            """
            n_hits = 32
            top = 5
            start = time.time()
            scores, retrieved_examples = ds.get_nearest_examples_batch('embeddings', question_embedding, k=n_hits)
            end = time.time()
            print('time:', end - start)
            return scores, retrieved_examples

        def cross_encoder_ranking(self, queries, retrieved_examples, top=10):
            """_summary_

            Args:
                queries (_type_): _description_
                retrieved_examples (_type_): _description_
                top (int, optional): _description_. Defaults to 10.
            """
            with open('evaluation_queries.txt', 'a') as f:
                for q in range(len(queries)):
                    cross_inp = [[queries[q], paragraph] for paragraph in retrieved_examples[q]['text']]
                    cross_scores = self.cross_encoder.predict(cross_inp)
                    sorted_indices = np.argsort(cross_scores)[::-1]

                    pos = self.hn_df[self.hn_df.qid == self.q_ids[indx][q]]['pos'].values[0]
                    pos_txt = self.corpus_df[self.corpus_df['_id'] == pos[0]]['text'].values[0]

                    rank = np.where(np.array(retrieved_examples[q]['_id'])[sorted_indices] == pos[0])
                    rank = rank[0] if len(rank) == 1 else -1

                    rank_BI = np.where(np.array(retrieved_examples[q]['_id']) == pos[0])
                    rank_BI = rank_BI[0] if len(rank_BI[0]) == 1 else -1

                    f.write('QUERY: ' + queries[q] + '\n')
                    f.write('RANK: ' + str(rank) + '\n')
                    f.write('RANK_BI: ' + str(rank_BI) + '\n')
                    f.write('ranking indices: ' + str(sorted_indices) + '\n')
                    f.write('POSITIVE: ' + pos[0] + ' --- ' + pos_txt + '\n\n')

                    for idx in sorted_indices[:top]:
                        f.write(
                            'TITLE: ' + retrieved_examples[q]['_id'][idx] + retrieved_examples[q]['title'][idx] + '\n'
                        )
                        f.write("\t{:.3f}\t{}\n\n".format(cross_scores[idx], retrieved_examples[q]['text'][idx]))
                        #print('CORPUS_ID', hit['corpus_id'])
                    f.write('---------------\n')

        def embed_queries(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            queries = self.qgen_df.text.values
            self.q_ids = self.qgen_df._id.values
            q_embeddings = self.bi_encoder.encode(queries, convert_to_tensor=False, show_progress_bar=True)
            return queries, q_embeddings

    dir_emb = dataset_hf.path
    ds = load_from_disk(os.path.join(dir_emb, "ds_with_embedding.hf"))

    # fais_index = glob.glob('{}/**.faiss'.format(dir_emb))[0]
    # print(fais_index)
    #ds.load_faiss_index('embeddings', os.path.join(dir_emb,fais_index.rsplit('/',1)[-1]))
    ds.add_faiss_index('embeddings', metric_type=faiss.METRIC_INNER_PRODUCT)
    print('ds loaded from memory')

    QE = QueryEvaluator(generated_queries.path, bi_encoder)
    queries, q_embs = QE.embed_queries()
    print(len(queries), q_embs.shape)
    #question_embeddings = bi_encoder.batch_encode_plus(queries, convert_to_tensor=False)

    n = 100  # number of queries to test on and print out
    import random
    random.seed(200)
    indx = random.sample(range(0, len(queries)), n)

    scores, retrieved_examples = QE.search_faiss_bi_encoder(ds, q_embs[indx])
    QE.cross_encoder_ranking(queries[indx], retrieved_examples, top=5)

    print('Uploading results to storage ...')
    storage_client = storage.Client()
    bucket = storage_client.bucket('semantic_search_blobstore')
    remote_path = os.path.join(
        'corpus_ds_with_embedding/evaluation',
        dataset_hf.path.rsplit('/')[-2], bi_encoder, 'evaluation_queries4.txt'
    )
    blob = bucket.blob(remote_path)
    blob.upload_from_filename('evaluation_queries.txt')
