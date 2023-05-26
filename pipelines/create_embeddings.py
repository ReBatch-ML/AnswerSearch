"""Pipeline to create embeddings for the preprocessed copora"""
from azureml.core import Environment, Experiment, Datastore, Dataset
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from packages.azureml_functions import get_ws, configure_computes
import argparse
import yaml


def get_pipeline_steps(ws, clusters, args):
    """
    Function to get the steps for the pipeline

    :param ws: AzureML workspace
    :param clusters: List of clusters to use
    :param args: Arguments from the command line

    :return: List of steps for the pipeline
    """
    steps = []

    datastore = Datastore.get(ws, 'workspaceblobstore')

    pipeline_param_batch_size = PipelineParameter(name="batch_size", default_value=args.batch_size)
    pipeline_param_max_length = PipelineParameter(name="max_length", default_value=args.max_length)
    pipeline_param_split_length = PipelineParameter(name="split_length", default_value=args.split_length)
    pipeline_param_extra_margin = PipelineParameter(name="extra_margin", default_value=args.extra_margin)
    pipeline_param_model_name = PipelineParameter(name="model_name", default_value=args.model_name)
    pipeline_param_model_version = PipelineParameter(name="model_version", default_value=args.model_version)

    datastore_paths = [(datastore, args.dataset)]
    files = Dataset.File.from_files(path=datastore_paths)

    # # Pipeline parameters
    # dataset = PipelineParameter(name="dataset", default_value=files)

    # ENVIRONMENT
    pipeline_pre_run_config = RunConfiguration()
    env = Environment.from_conda_specification(
        name='pipeline_environment_testing', file_path='environments/create_embedding_environment.yml'
    )

    # env.docker.base_image = "mcr.microsoft.com/azureml/curated/acpt-pytorch-1.13-py38-cuda11.7-gpu:1"

    pipeline_pre_run_config.environment = env

    step_inputs = []

    command_args = [
        '--dataset',
        files.as_download(),
        '--batch_size',
        pipeline_param_batch_size,
        '--max_length',
        pipeline_param_max_length,
        '--split_length',
        pipeline_param_split_length,
        '--extra_margin',
        pipeline_param_extra_margin,
    ]

    command_args.append('--model_name')
    command_args.append(pipeline_param_model_name)
    command_args.append('--model_version')
    command_args.append(pipeline_param_model_version)

    command_args.append("--save_on_azure")

    # STEP CONFIG
    batch_step = PythonScriptStep(
        name="create embeddings",
        source_directory='src/models',
        script_name="embedding_creator.py",
        inputs=step_inputs,
        arguments=command_args,
        compute_target=clusters[0],
        runconfig=pipeline_pre_run_config,
        allow_reuse=True
    )
    steps.append(batch_step)

    return steps


def main():
    """
    Main function to run the pipeline
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default='dev')
    args = parser.parse_args()

    ws = get_ws(args.stage)

    # OR REPLACE WITH TRAINED MODEL
    # read yaml files to determine the default model for creating embeddings
    with open("configs/deployment_config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        language = cfg["corpus_language"].lower()

    # read yaml files to determine which model is used for the selected language
    with open("configs/model_languages.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        default_bi_encoder = cfg[language]["bi_encoder"]

    # CONFIGUREABLE PARAMETERS

    # batch size for the embedding creation
    args.batch_size = 64
    # The maximum length of the input
    args.max_length = 512
    # The length to split the input into (-2 because of adding the cls and sep tokens)
    args.split_length = 510
    # The extra margin to add to the split length
    args.extra_margin = 20
    # The name of the model to use
    args.model_name = default_bi_encoder
    # The version of the model to use
    args.model_version = -1

    # location of the preprocessed dataset
    args.dataset = "processed"

    # clusters = [("NC6", "Standard_NC6", 2)]
    clusters = [("A100", "Standard_NC24ads_A100_v4", 1)]
    aml_computes = configure_computes(ws, clusters=clusters)

    pipeline_steps = get_pipeline_steps(ws, aml_computes, args)
    pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
    print("Pipeline is built.")

    exp_name = "CreateEmbeddings"
    exp = Experiment(workspace=ws, name=exp_name)

    pipeline_run = exp.submit(pipeline)
    pipeline_run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()
