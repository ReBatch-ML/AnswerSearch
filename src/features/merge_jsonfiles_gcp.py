'''
Mini pipeline on GCP that merges different jsonl files
(was used to speed up preprocessing by preprocessing differnt years in parallel and afterwards putting everything together)
'''

import kfp
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, InputPath, Dataset
from google.cloud import aiplatform
from google.oauth2 import service_account
from src.features.paragraph_splitting import paragraph_generation
from google.cloud import storage
from kfp.v2.dsl import importer
from typing import List, Union
import yaml

# project information and save location
BUCKET_NAME_SS = 'gs://semantic_search_blobstore'
BUCKET_NAME_GSC = 'gs://gsc-migrated-blobstore'

PROJECT = 'rebatch-sandbox-329013'
PIPELINE_ROOT = f"{BUCKET_NAME_SS}/pipeline_root_ss/"
LOCATION = 'europe-west1'
# GPU = 'NVIDIA_TESLA_A100'  #A100 only works  in europe-west4


def merge_gcp_jsonlfiles(gcp_dir: Input[Dataset], output_filename: str):
    """_summary_

    Args:
        gcp_dir (Input[Dataset]): _description_
        output_filename (str): _description_
    """
    import glob
    import jsonlines

    source_dir = gcp_dir.path
    source_files = glob.glob(f'{source_dir}/**.jsonl')
    source_files = [s for s in source_files if '_PART_' in s]
    print(source_files)

    with jsonlines.open(f'{source_dir}/{output_filename}.jsonl', mode='w') as writer:
        for source_path in source_files:
            print(source_path)
            with jsonlines.open(source_path) as reader:
                for line in reader:
                    writer.write(line)


# authenticate to ai platform
credentials = service_account.Credentials.from_service_account_file('.cloud/.gcp/GCP_SERVICE_ACCOUNT.json')
aiplatform.init(
    project=PROJECT,
    location=LOCATION,
    staging_bucket=BUCKET_NAME_SS,
    credentials=credentials,
)


def compile_pipeline():
    """_summary_

    Returns:
        _type_: _description_
    """
    artifact_uri = f"gs://semantic_search_blobstore/preprocessed_corpus"  #' f"gs://semantic_search_blobstore/{args['corpus_path_on_datastore']}"
    template_path = "merge-gcp-files.json"

    # preprocess data component
    merge_files = kfp.components.create_component_from_func_v2(
        merge_gcp_jsonlfiles,
        base_image='python:3.8',  #europe-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-11:latest', # Optional
        packages_to_install=['jsonlines']
    )

    @dsl.pipeline(
        # Default pipeline root. You can override it when submitting the pipeline.
        pipeline_root=PIPELINE_ROOT,
        # A name for the pipeline. Use to determine the pipeline Context.
        name="merge-files",
    )
    def pipeline(output_filename: str):
        """_summary_

        Args:
            output_filename (str): _description_
        """

        imp = kfp.v2.dsl.importer(
            artifact_uri=artifact_uri,
            artifact_class=Dataset,
            reimport=True,
        )

        print(artifact_uri)

        prep_task = merge_files(imp.output, output_filename)
        prep_task.set_cpu_limit('8')

    compiler.Compiler().compile(pipeline_func=pipeline, package_path=template_path)

    return template_path


###### STATE PARAMS AND START PIPELINE RUN #####

ds_name = 'corpus_all_length_250'

template_path = compile_pipeline()

job = aiplatform.PipelineJob(
    display_name='merge-files',
    template_path=template_path,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={'output_filename': ds_name}
)

job.run()
