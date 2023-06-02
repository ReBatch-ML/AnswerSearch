'''
Embedding of the corpus executed by pipelines/gcp_embed_corpus
takes as input:
- preprocessed paragraphs pathname (relative to gs://semantic_search_blobstore/preprocessed_corpus/)
- bi-encoder model (trained or untrained)
- evaluation, if True: assume we are dealing with test data and upload to GCP storage instead of Azure for further evaluation
- prepend; if True prepend title to paragraph before making the embedding
'''
from kfp.v2.dsl import InputPath, Dataset, Input


def embed_corpus(
    corpus_filepath: Input[Dataset],
    bi_encoder: str = 'multi-qa-MiniLM-L6-cos-v1',
    evaluation: bool = True,
    prepend: bool = False
):
    """
    Function that defines the entire step/

    Args:
        corpus_filepath (Input[Dataset]): _description_
        bi_encoder (str, optional): _description_. Defaults to 'multi-qa-MiniLM-L6-cos-v1'.
        evaluation (bool, optional): _description_. Defaults to True.
        prepend (bool, optional): _description_. Defaults to False.

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """

    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    # from azureml.core import Environment, Experiment, Datastore, Run
    from azureml.core import Dataset as AzureDataset
    from azureml.data.datapath import DataPath
    from azureml.core import Model, Workspace
    from azureml.core.authentication import ServicePrincipalAuthentication
    from google.cloud import secretmanager
    from google.cloud import storage

    import os
    import json
    import glob
    import pandas as pd
    from datasets import Dataset
    import datasets
    import argparse
    import math

    # def embed_corpus(corpus_filepath: Input[Dataset],bi_encoder: str = 'multi-qa-MiniLM-L6-cos-v1'):
    # def embed_corpus(corpus_filepath, bi_encoder='multi-qa-MiniLM-L6-cos-v1'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    def prepend_title(line):
        """
        Prepend title to paragraph

        Args:
            line (_type_): _description_

        Returns:
            _type_: _description_
        """
        title = line['title']
        text = title + ': ' + line['text']
        return text

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

    class CorpusEmbedder:
        """
        Class that handles the embedding of corpora, the only thing needed for Azure, where GCP needs a function wrapped around this
        """

        def __init__(self, bi_encoder_name, corpus_filepath, prepend=False):
            print('initialising CorpusEmbedder')
            try:
                bi_encoder_name, version = bi_encoder_name.split('_version_')
                model = Model(workspace=ws, name=bi_encoder_name, version=int(version)).download(exist_ok=True)
                bi_encoder = SentenceTransformer(model)
                print(f'Model {bi_encoder_name} version {version} loaded from ws')
            except Exception as e:
                print(e)
                print(f'Loading pretrained {bi_encoder_name} from Sentence-Transformers')
                bi_encoder = SentenceTransformer(bi_encoder_name)

            bi_encoder.max_seq_length = 512  #Truncate long passages to 512 tokens
            bi_encoder.tokenizer.truncate_sequences = False
            self.tokenizer = bi_encoder.tokenizer
            self.bi_encoder = bi_encoder

            data = []
            self.passages = []
            file_path = glob.glob('{}'.format(corpus_filepath), recursive=True)[0]
            #self.corpus_df = pd.read_json(path_or_buf=f'{file_path}', lines=True)

            print('args', corpus_filepath)
            print(f'loading files, prepend title= {prepend}')
            with open(file_path) as f:
                for line in f:
                    line = json.loads(line)
                    data.append(line)
                    if prepend:
                        self.passages.append(prepend_title(line))
            self.corpus_df = pd.DataFrame(data)
            if not prepend:
                self.passages = self.corpus_df['text'].values

            self.output_dir = 'outputs'
            os.makedirs(self.output_dir, exist_ok=True)

        def upload_local_directory_to_gcs(self, local_path, bucket, gcs_path):
            """_summary_

            Args:
                local_path (_type_): _description_
                bucket (_type_): _description_
                gcs_path (_type_): _description_
            """
            assert os.path.isdir(local_path)
            for local_file in glob.glob(local_path + '/**'):
                if not os.path.isfile(local_file):
                    self.upload_local_directory_to_gcs(
                        local_file, bucket, gcs_path + "/" + os.path.basename(local_file)
                    )
                else:
                    remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
                    blob = bucket.blob(remote_path)
                    blob.upload_from_filename(local_file)

        def embed_corpus(
            self,
            batch_size=128,
        ):
            """_summary_

            Args:
                batch_size (int, optional): _description_. Defaults to 128.

            Returns:
                _type_: _description_
            """
            corpus_embeddings = self.bi_encoder.encode(
                self.passages, batch_size=batch_size, device=device, convert_to_numpy=True, show_progress_bar=True
            )

            n_chunks = 10  # Dataset.from_dict crashes if corpus_embeddings too big, so split in chunks and afterwards concatenate
            chunks = len(corpus_embeddings) // n_chunks
            dset_embed = Dataset.from_dict({"embeddings": corpus_embeddings[:chunks]})
            for c in range(1, n_chunks + 1):
                print(c)
                d = Dataset.from_dict({"embeddings": corpus_embeddings[c * chunks:(c+1) * chunks]})
                dset_embed = datasets.concatenate_datasets([dset_embed, d], axis=0)

            # build dataset with all original info on paragraphs together with the embeddings
            dset_passages = Dataset.from_pandas(self.corpus_df)
            self.ds_with_embeddings = datasets.concatenate_datasets([dset_passages, dset_embed], axis=1)

            self.ds_with_embeddings.save_to_disk(self.output_dir + "/ds_with_embedding.hf")
            return len(self.corpus_df)

        def add_faiss_index(self, index_string='IVF4096,Flat'):
            """_summary_

            Args:
                index_string (str, optional): _description_. Defaults to 'IVF4096,Flat'.
            """
            print('Training FAISS index')
            self.ds_with_embeddings.add_faiss_index(
                column='embeddings', faiss_verbose=True, string_factory=index_string, train_size=len(self.corpus_df)
            )  #, index_name='IVF4096')
            print('Saving FAISS index')
            self.ds_with_embeddings.save_faiss_index(
                'embeddings', self.output_dir + '/' + index_string.replace(',', '_') + '.faiss'
            )

        def upload_corpus_and_index(self, datastore, folder_name, evaluation):
            """_summary_

            Args:
                datastore (_type_): _description_
                folder_name (_type_): _description_
                evaluation (_type_): _description_
            """
            if evaluation:
                print('Evaluation embeddings saved to GCP')
                storage_client = storage.Client()
                bucket = storage_client.bucket('semantic_search_blobstore')
                filename = corpus_filepath.path.rsplit('/', 3)[-3] + '_eval'
                self.upload_local_directory_to_gcs(
                    local_path=self.output_dir,
                    bucket=bucket,
                    gcs_path=f'corpus_ds_with_embedding/evaluation/{filename}/{bi_encoder}'
                )

            else:
                print('Uploading to Azure ...')
                AzureDataset.File.upload_directory(
                    src_dir=self.output_dir, target=DataPath(
                        datastore,
                        folder_name,
                    ), overwrite=True
                )

    # initialise Embedder
    try:
        embedder = CorpusEmbedder(bi_encoder, corpus_filepath.path, prepend)
        corpus_filename = corpus_filepath.path.split('.')[0].split('/')[-1]
        print('filename', corpus_filename)
        output_folder = f'corpus_ds_with_embedding/{corpus_filename}_{bi_encoder}'
        # create embeddings
        corpus_size = embedder.embed_corpus(batch_size=256)
        # # train index and save
        # string_factory = f"IVF{int(4*math.sqrt(corpus_size))},Flat"
        # embedder.add_faiss_index(index_string = string_factory)
        embedder.upload_corpus_and_index(ws.get_default_datastore(), output_folder, evaluation)
    except Exception as e:
        print(e)
        print(type(e))
        raise e


# DEPRECATED
# Used to work with run_embed_corpus pipeline on Azure
# now embed_corpus definition is used directly in gcp pipeline
if __name__ == "__main__":
    # Get parameters
    run = Run.get_context()
    ws = run.experiment.workspace

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--corpus_filepath',
        type=str,
    )
    args = parser.parse_args()

    # initialise Embedder
    bi_encoder = 'multi-qa-MiniLM-L6-cos-v1'
    embedder = CorpusEmbedder(bi_encoder, args.corpus_filepath)
    corpus_filename = args.corpus_filepath.split('.')[0].split('/')[-1]
    output_folder = f'corpus_ds_with_embedding/{corpus_filename}'

    # create embeddings
    corpus_size = embedder.embed_corpus(batch_size=256)

    # train index and save
    string_factory = f"IVF{int(4*math.sqrt(corpus_size))},Flat"
    # string_factory = f"IVF65536_HNSW32{int(4*sqrt(corpus_size))},Flat"
    embedder.add_faiss_index(index_string=string_factory)
    embedder.upload_corpus_and_index(ws.get_default_datastore(), output_folder)
