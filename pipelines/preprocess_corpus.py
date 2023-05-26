"""Pipeline to preprocess the raw corpus data so it can be used to create embeddings"""
from azureml.core import Environment, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
import yaml
from packages.azureml_functions import get_ws, configure_computes
import argparse


def main():
    """
    Main function to create and run the pipeline
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait_for_completion", action='store_true', default=True)
    parser.add_argument("--stage", type=str, default="dev", help="Deployment stage")
    args = parser.parse_args()

    ws = get_ws(args.stage)

    # What type of compute you need
    clusters = [('DS12-cpu', 'Standard_DS12_v2', 1)]
    compute = configure_computes(ws, clusters)

    # Make a conda env from specifications
    preprocess_env = Environment.from_conda_specification('preprocess_env', 'environments/preprocess_environment.yml')
    preprocess_run_config = RunConfiguration()
    preprocess_run_config.environment = preprocess_env

    # ARGUMENTS

    # read raw data path from config yaml
    with open("configs/deployment_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
        data_path = config['raw_corpus_path']
        output_file_name = config['corpora'][0]['location']

    datastore = Datastore.get(ws, 'workspaceblobstore')
    datastore_paths = [(datastore, data_path)]
    files = Dataset.File.from_files(path=datastore_paths)

    arguments = ['--dataset', files.as_download(), '--output_file_name', output_file_name]

    #step configuration
    NAME = "preprocess_data"
    prep_step = PythonScriptStep(
        name=NAME,
        source_directory='src/datasets',
        script_name="preprocess.py",
        inputs=[],
        arguments=arguments,
        compute_target=compute[0],
        runconfig=preprocess_run_config,
        allow_reuse=False
    )

    pipeline_steps = [prep_step]
    preprocess_pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

    print('submitting pipeline')

    pipeline_run = preprocess_pipeline.submit(experiment_name=NAME, pipeline_parameters={})
    if args.wait_for_completion:
        pipeline_run.wait_for_completion()


if __name__ == "__main__":
    main()
