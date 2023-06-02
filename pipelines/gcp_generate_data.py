'''
Pipeline to generate synthetic GPL data on GCP
Most important parameters can be found and changed at GPL/gpl/gpl_args.yml
'''
import kfp
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, InputPath, Dataset
from google.cloud import aiplatform
from google.oauth2 import service_account
from GPL.gpl.training_data import generate_gpl_data
from google.cloud import storage
from kfp.v2.dsl import importer
from typing import List, Union
import yaml

# project information and save location
BUCKET_NAME_SS = 'gs://semantic_search_blobstore'
PROJECT = 'rebatch-sandbox-329013'
PIPELINE_ROOT = f"{BUCKET_NAME_SS}/pipeline_root_ss/"
LOCATION = 'europe-west4'
GPU = 'NVIDIA_TESLA_A100'  #A100 only works  in europe-west4

# authenticate to ai platform
credentials = service_account.Credentials.from_service_account_file('.cloud/.gcp/GCP_SERVICE_ACCOUNT.json')
aiplatform.init(
    project=PROJECT,
    location=LOCATION,
    staging_bucket=BUCKET_NAME_SS,
    credentials=credentials,
)


def get_gpl_args():
    """
    Function that will read the arguments set in the GPL folder

    Returns:
        Dict: GPL arguments 
    """
    with open("src/models/GPL/gpl/gpl_args.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    args_gen = args['datagen_args']
    args_shared = args['shared_args']

    DATASET_NAME = args_shared['corpus_name']
    path_on_datastore = f'preprocessed_corpus/{DATASET_NAME}.jsonl'
    del args_shared['corpus_name']
    args_shared.update(
        {
            'corpus_path_on_datastore': path_on_datastore,
            'path_to_generated_data': f'/gcs/semantic_search_blobstore/gpl_training_data/{DATASET_NAME}/all'
        }
    )

    gpl_args = dict(**args_gen, **args_shared)
    return gpl_args


def compile_pipeline(args):
    """
    Compile the pipeline definition so that its template can be run later as a job.

    Args:
        args (Dict): arguments needed to configure the pipeline steps

    Returns:
        str: template path where the pipeline template is stored.
    """
    artifact_uri = f"gs://semantic_search_blobstore/{args['corpus_path_on_datastore']}"
    template_path = "gpl-generate-data.json"

    # preprocess data component
    generate_data_gpl = kfp.components.create_component_from_func_v2(
        generate_gpl_data,
        base_image=
        'europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest',  #europe-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-11:latest', # Optional
        packages_to_install=['pandas', 'gpl', 'jsonlines', 'google-cloud-storage']
    )

    @dsl.pipeline(
        # Default pipeline root. You can override it when submitting the pipeline.
        pipeline_root=PIPELINE_ROOT,
        # A name for the pipeline. Use to determine the pipeline Context.
        name="gpl-generate-data",
    )
    def pipeline(
        path_to_generated_data: str,
        evaluation_output: str = 'output',
        qgen_prefix: str = 'qgen',
        base_ckpt: str = 'distilbert-base-uncased',
        generator: str = 'BeIR/query-gen-msmarco-t5-base-v1',
        cross_encoder: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        batch_size_gpl: int = 32,
        batch_size_generation: int = 32,
        max_seq_length: int = 350,
        queries_per_passage: int = 3,
        gpl_steps: int = 140000,
        use_amp: bool = False,
        retrievers: List[str] = ['msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3'],
        retriever_score_functions: List[str] = ['cos_sim', 'cos_sim'],
        negatives_per_query: int = 50,
        eval_split: str = 'test',
        use_train_qrels: bool = False,
        gpl_score_function: str = 'dot',
        corpus_path_on_datastore: str = ''
    ):
        """
        Function that actually defines the pipeline itself.

        Args:
            path_to_generated_data (str): _description_
            evaluation_output (str, optional): _description_. Defaults to 'output'.
            qgen_prefix (str, optional): _description_. Defaults to 'qgen'.
            base_ckpt (str, optional): _description_. Defaults to 'distilbert-base-uncased'.
            generator (str, optional): _description_. Defaults to 'BeIR/query-gen-msmarco-t5-base-v1'.
            cross_encoder (str, optional): _description_. Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
            batch_size_gpl (int, optional): _description_. Defaults to 32.
            batch_size_generation (int, optional): _description_. Defaults to 32.
            max_seq_length (int, optional): _description_. Defaults to 350.
            queries_per_passage (int, optional): _description_. Defaults to 3.
            gpl_steps (int, optional): _description_. Defaults to 140000.
            use_amp (bool, optional): _description_. Defaults to False.
            retrievers (List[str], optional): _description_. Defaults to ['msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3'].
            retriever_score_functions (List[str], optional): _description_. Defaults to ['cos_sim', 'cos_sim'].
            negatives_per_query (int, optional): _description_. Defaults to 50.
            eval_split (str, optional): _description_. Defaults to 'test'.
            use_train_qrels (bool, optional): _description_. Defaults to False.
            gpl_score_function (str, optional): _description_. Defaults to 'dot'.
            corpus_path_on_datastore (str, optional): _description_. Defaults to ''.
        """
        imp = kfp.v2.dsl.importer(
            artifact_uri=artifact_uri,
            artifact_class=Dataset,
            reimport=False,
        )

        print(artifact_uri)

        datagen_task = generate_data_gpl(
            path_to_generated_data=path_to_generated_data,
            output_dir=PIPELINE_ROOT,
            evaluation_output=evaluation_output,
            qgen_prefix=qgen_prefix,
            base_ckpt=base_ckpt,
            generator=generator,
            cross_encoder=cross_encoder,
            batch_size_gpl=batch_size_gpl,
            batch_size_generation=batch_size_generation,
            max_seq_length=max_seq_length,
            queries_per_passage=queries_per_passage,
            gpl_steps=gpl_steps,
            use_amp=use_amp,
            retrievers=retrievers,
            retriever_score_functions=retriever_score_functions,
            negatives_per_query=negatives_per_query,
            eval_split=eval_split,
            use_train_qrels=use_train_qrels,
            gpl_score_function=gpl_score_function,
            corpus_path_on_datastore=imp.output,
        )
        datagen_task.add_node_selector_constraint('cloud.google.com/gke-accelerator', GPU).set_gpu_limit(1)

    compiler.Compiler().compile(pipeline_func=pipeline, package_path=template_path)

    return template_path


###### STATE PARAMS AND START PIPELINE RUN #####

gpl_args = get_gpl_args()
template_path = compile_pipeline(gpl_args)

job = aiplatform.PipelineJob(
    display_name='gsc-gpl_generate_data',
    template_path=template_path,
    pipeline_root=PIPELINE_ROOT,
    parameter_values=gpl_args
)

job.run()
