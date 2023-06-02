""" GCP preprocessing pipeline. """
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


def get_dependencies():
    """
    Retrieve dependencies from environments/preprocess_paragraphs_environment.yaml file.

    Returns:
        tuple: python version, packages to be installed by conda, packages installed via pip
    """
    with open("environments/preprocess_paragraphs_environment.yaml") as f:
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


def compile_pipeline():
    """
    Compile the pipeline definition so that its template can be run later as a job.

    Args:
        args (Dict): arguments needed to configure the pipeline steps

    Returns:
        str: template path where the pipeline template is stored.
    """
    artifact_uri = f"gs://gsc-migrated-blobstore/raw/publicregister/**/EN"  #' f"gs://semantic_search_blobstore/{args['corpus_path_on_datastore']}"
    template_path = "paragraph-generation.json"

    # preprocess data component
    generate_paragraphs = kfp.components.create_component_from_func_v2(
        paragraph_generation,
        base_image='python:3.8',  #europe-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-11:latest', # Optional
        packages_to_install=pip_deps  #+["google-cloud-storage"]
    )

    @dsl.pipeline(
        # Default pipeline root. You can override it when submitting the pipeline.
        pipeline_root=PIPELINE_ROOT,
        # A name for the pipeline. Use to determine the pipeline Context.
        name="generate-paragraphs",
    )
    def pipeline(output_filename: str, years: List[int]):
        """
        Pipeline definition.

        Args:
            output_filename (str): _description_
            years (List[int]): _description_
        """
        imp = kfp.v2.dsl.importer(
            artifact_uri=artifact_uri,
            artifact_class=Dataset,
            reimport=True,
        )

        print(artifact_uri)

        prep_task = generate_paragraphs(imp.output, output_filename, years)
        prep_task.set_cpu_limit('8')

    compiler.Compiler().compile(pipeline_func=pipeline, package_path=template_path)

    return template_path


###### STATE PARAMS AND START PIPELINE RUN #####

#YEARS =[year for year in range(1995,2023)] # years for which documents will be processed
YEARS = [year for year in range(1993, 2023)]
ds_name = 'corpus_' + str(min(YEARS)) + '-' + str(max(YEARS)) + '_length_250'
#ds_name = 'corpus_all_length_250'

template_path = compile_pipeline()

job = aiplatform.PipelineJob(
    display_name='gsc-gpl_generate_data',
    template_path=template_path,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        'years': YEARS,
        'output_filename': ds_name
    }
)

job.run()
