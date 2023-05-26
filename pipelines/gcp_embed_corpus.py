'''
Pipeline to embed corpus with params on GCP
BI_ENCODER 
CORPUS_NAME (name without extension coming from gs://semantic_search_blobstore/preprocessed_corpus/)
EVALUATION, if True, only embed training+test data and afterwards perform evaluation step on this dataset
PREPEND_TITLE if True, prepend title of document to each paragraph when making the embedding
'''

import kfp
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, InputPath, Dataset
from google.cloud import aiplatform
from google.oauth2 import service_account
from src.features.embed_corpus import embed_corpus
from src.features.evaluate_corpus import evaluate_corpus
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

client = storage.Client.from_service_account_json(json_credentials_path='.cloud/.gcp/GCP_SERVICE_ACCOUNT.json')
bucket = storage.Bucket(client, BUCKET_NAME_SS[5:])


def get_dependencies():
    """
    Retrieve dependencies from environments/embed_corpus_environment.yaml file.

    Returns:
        tuple: python version, packages to be installed by conda, packages installed via pip
    """
    with open("environments/embed_corpus_environment.yaml") as f:
        deps = yaml.load(f, Loader=yaml.FullLoader)

    python_version = deps["dependencies"][0].split('=')[-1]
    conda_packages = deps["dependencies"][1:-1]
    pip_packages = deps["dependencies"][-1]['pip']

    return python_version, conda_packages, pip_packages


python_version, _, pip_deps = get_dependencies()

# authenticate to ai platform
credentials = service_account.Credentials.from_service_account_file('.cloud/.gcp/GCP_SERVICE_ACCOUNT.json')
aiplatform.init(
    project=PROJECT,
    location=LOCATION,
    staging_bucket=BUCKET_NAME_SS,
    credentials=credentials,
)


def compile_pipeline(args):
    """
    Compile the pipeline definition so that its template can be run later as a job.

    Args:
        args (Dict): arguments needed to configure the pipeline steps

    Returns:
        str: template path where the pipeline template is stored.
    """
    if args['evaluation']:
        artifact_uri = f"gs://semantic_search_blobstore/gpl_training_data/{args['corpus_name']}/test/corpus.jsonl"
        ds_hf_uri = f"gs://semantic_search_blobstore/corpus_ds_with_embedding/evaluation/{args['corpus_name']}_eval/{args['bi_encoder']}"
        check_uri = f"corpus_ds_with_embedding/evaluation/{args['corpus_name']}_eval/{args['bi_encoder']}/ds_with_embedding.hf/dataset.arrow"
        queries_uri = f"gs://semantic_search_blobstore/gpl_training_data/{args['corpus_name']}"
        print('INFO: evaluating on', artifact_uri, '\n')
    else:
        artifact_uri = f"gs://semantic_search_blobstore/preprocessed_corpus/{args['corpus_name']}.jsonl"  #' f"gs://semantic_search_blobstore/{args['corpus_path_on_datastore']}"
        print('INFO: creating embeddings on ', artifact_uri)

    template_path = "embed-corpus.json"

    # preprocess data component
    generate_paragraphs = kfp.components.create_component_from_func_v2(
        embed_corpus,
        base_image='europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest',  # Optional
        packages_to_install=pip_deps  #+["google-cloud-storage"]
    )

    evaluate = kfp.components.create_component_from_func_v2(
        evaluate_corpus,
        #base_image='python:3.8',
        base_image='europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest',  # Optional
        packages_to_install=[
            'pandas', 'azureml-defaults==1.44', 'joblib', 'datasets', 'sentence-transformers', 'numpy>=1.18.5',
            'faiss-gpu', 'google-cloud-storage', 'google-cloud-secret-manager'
        ]
    )

    @dsl.pipeline(
        # Default pipeline root. You can override it when submitting the pipeline.
        pipeline_root=PIPELINE_ROOT,
        # A name for the pipeline. Use to determine the pipeline Context.
        name="embed-corpus",
    )
    def pipeline(corpus_name: str, bi_encoder: str, evaluation: bool, prepend: bool):
        """
        Function that actually defines the pipeline itself.

        Args:
            corpus_name (str): _description_
            bi_encoder (str): _description_
            evaluation (bool): _description_
            prepend (bool): _description_
        """
        import_preprocessed = kfp.v2.dsl.importer(
            artifact_uri=artifact_uri,
            artifact_class=Dataset,
            reimport=True,
        )

        if not args['evaluation']:
            embed_task = generate_paragraphs(
                import_preprocessed.output, bi_encoder, args['evaluation'], args['prepend']
            )
            embed_task.add_node_selector_constraint('cloud.google.com/gke-accelerator', GPU).set_gpu_limit(1)

        else:
            import_queries = kfp.v2.dsl.importer(
                artifact_uri=queries_uri,
                artifact_class=Dataset,
                reimport=True,
            )
            import_dataset = kfp.v2.dsl.importer(
                artifact_uri=ds_hf_uri,
                artifact_class=Dataset,
                reimport=True,
            )

            if bucket.blob(check_uri).exists():
                print('INFO: Evaluation embeddings already exist, only doing evaluation\n')
                evaluate_task = evaluate(bi_encoder, import_queries.output, import_dataset.output)
                evaluate_task.add_node_selector_constraint('cloud.google.com/gke-accelerator', GPU).set_gpu_limit(1)
                evaluate_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
            else:
                print('INFO: Creating embeddings first and afterwards do evaluation\n')
                embed_task = generate_paragraphs(import_preprocessed.output, bi_encoder, args['evaluation'])
                embed_task.add_node_selector_constraint('cloud.google.com/gke-accelerator', GPU).set_gpu_limit(1)
                #with dsl.Condition(embed_task.output=="done"):
                evaluate_task = evaluate(bi_encoder, import_queries.output, import_dataset.output)
                evaluate_task.add_node_selector_constraint('cloud.google.com/gke-accelerator',
                                                           GPU).set_gpu_limit(1).after(embed_task)

        # print(artifact_uri)

        # #if not args['evaluate_only']:
        # embed_task = generate_paragraphs(import_preprocessed.output, bi_encoder)
        # embed_task.add_node_selector_constraint('cloud.google.com/gke-accelerator', GPU).set_gpu_limit(1)

    compiler.Compiler().compile(pipeline_func=pipeline, package_path=template_path)

    return template_path


###### STATE PARAMS AND START PIPELINE RUN #####
BI_ENCODER = 'multi-qa-mpnet-base-dot-v1_TRAINED_2022-10-21-14-33-13_version_45'  #_TRAINED_2022-10-20-18-30-18_version_14'#_TRAINED_2022-10-16-17-50-22_version_6'#_TRAINED_2022-10-13-09-35-19_version_42' #_TRAINED_2022-10-05-07-21-14_version_14' #_2022-09-22-12-43-41'
CORPUS_NAME = 'corpus_all_length_250'  # name without extension (coming from gs://semantic_search_blobstore/preprocessed_corpus/)
EVALUATION = False
PREPEND_TITLE = True

embedding_args = {
    'corpus_name': CORPUS_NAME,
    'bi_encoder': BI_ENCODER,
    'evaluation': EVALUATION,
    'prepend': PREPEND_TITLE
}

template_path = compile_pipeline(embedding_args)

job = aiplatform.PipelineJob(
    display_name='embed-corpus',
    template_path=template_path,
    pipeline_root=PIPELINE_ROOT,
    parameter_values=embedding_args
)

job.run(service_account='serice-principal-testing@rebatch-sandbox-329013.iam.gserviceaccount.com')
