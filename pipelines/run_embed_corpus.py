'''
DEPRECATED
Used previously on Azure but latest work done in GCP pipelines
'''
from packages.azureml_functions import get_ws
from azureml.core import Environment, Dataset, Experiment, Datastore, Run
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineEndpoint
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep, PythonScriptStep
from azureml.data import OutputFileDatasetConfig, FileDataset
from azureml.core.runconfig import RunConfiguration
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data.datapath import DataPath, DataPathComputeBinding
import argparse
import os


def configure_computes(ws, clusters):
    '''
    clusters is a list consisting of the tuples (cluster_name, vm_size, max_nodes)
    e.g. cluster_names = [(cpu-cluster, STANDARD_D2_V2, 2), (gpu-cluster, Standard_NC6, 4)]
    '''
    for cluster_name, vm_size, max_nodes in clusters:
        # Verify that cluster does not exist already
        try:
            cluster = ComputeTarget(workspace=ws, name=cluster_name)
            vm_size_existing = cluster.serialize()['properties']['properties']['vmSize']
            if vm_size_existing.lower() != vm_size.lower():
                print(
                    f'WARNING: cluster {cluster_name} exists but with vm_size {vm_size_existing} instead of requested {vm_size} \nWe will still use the existing cluster'
                )
            else:
                print(f'Found existing cluster {cluster_name}, use it.')
        except ComputeTargetException:
            # To use a different region for the compute, add a location='<region>' parameter
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                max_nodes=max_nodes,
                idle_seconds_before_scaledown=300,
            )
            cluster = ComputeTarget.create(ws, cluster_name, compute_config)
            print(f"Creating new cluster {cluster_name} of type {vm_size} with {max_nodes} nodes")

        cluster.wait_for_completion(show_output=False)


if __name__ == "__main__":

    ws = get_ws()
    datastore = ws.get_default_datastore()
    #clusters = [('gsc-promo', 'Standard_NC6_PROMO', 1)]
    clusters = [('nc6-gpu', 'Standard_NC6', 1)]
    configure_computes(ws, clusters)
    embedding_env = Environment.from_conda_specification(
        'embed_corpus_env', 'environments/embed_corpus_environment.yaml'
    )
    embedding_run_config = RunConfiguration()
    embedding_run_config.environment = embedding_env

    input_data_loc = DataPath(
        datastore=datastore,
        path_on_datastore='preprocessed_corpus/preprocessed_corpus_staff_regulations_stafreg_other_servants.jsonl'
    )
    input_data = Dataset.File.from_files(path=input_data_loc)

    #step configuration
    corpus_embedding_step = PythonScriptStep(
        name="embed_corpus",
        source_directory='src/features',
        script_name="embed_corpus.py",
        inputs=[],
        arguments=['--corpus_filepath', input_data.as_download()],
        compute_target=clusters[0][0],
        runconfig=embedding_run_config,
        allow_reuse=False
    )

    pipeline_steps = [corpus_embedding_step]
    preprocess_pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

    print('submitting pipeline ')
    preprocess_pipeline.submit(experiment_name='Embed_corpus')
