'''
DEPRECATED
Used previously on Azure but latest work done in GCP pipelines
'''
from packages.azureml_functions import get_ws
from azureml.core import Environment, Dataset, Experiment, Datastore
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineEndpoint
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep, PythonScriptStep
from azureml.data import OutputFileDatasetConfig, FileDataset
from azureml.core.runconfig import RunConfiguration
# from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
# from azureml.data.datapath import DataPath, DataPathComputeBinding
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
    datastore = Datastore.get(ws, 'gsc_data')

    clusters = [('DS12-cpu', 'Standard_DS12_v2', 1)]
    configure_computes(ws, clusters)
    preprocess_env = Environment.from_conda_specification(
        'preprocess_env', 'environments/preprocess_paragraphs_environment.yaml'
    )
    preprocess_run_config = RunConfiguration()
    preprocess_run_config.environment = preprocess_env

    YEARS = [year for year in range(2021, 2022)]  # years for which documents will be processed
    ds_name = 'corpus_' + str(min(YEARS)) + '-' + str(max(YEARS))  #+ '_improved'

    datastore_paths_EN = [(datastore, f'raw/publicregister/{year}/EN/*.json') for year in YEARS]
    documents_EN = Dataset.File.from_files(path=datastore_paths_EN)

    #step configuration
    paragraph_splitting_step = PythonScriptStep(
        name="preprocess paragraphs",
        source_directory='src/features',
        script_name="paragraph_splitting.py",
        inputs=[],
        arguments=['--folder_files_EN', documents_EN.as_download(), '--output_filename', ds_name],
        compute_target=clusters[0][0],
        runconfig=preprocess_run_config,
        allow_reuse=False
    )

    pipeline_steps = [paragraph_splitting_step]
    preprocess_pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

    print('submitting pipeline ')
    preprocess_pipeline.submit(experiment_name='paragraph_splitting')
