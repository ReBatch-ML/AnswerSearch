'''
DEPRECATED
Used previously on Azure but latest work done in GCP pipelines
'''

from packages.azureml_functions import get_ws
from azureml.core import Environment, Dataset, Experiment, Datastore, Run
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineEndpoint
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.steps import PythonScriptStep
#from azureml.data import OutputFileDatasetConfig, FileDataset
from azureml.core.runconfig import RunConfiguration
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data.datapath import DataPath, DataPathComputeBinding
import argparse
import os
import yaml


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
    clusters = [('nc6-gpu', 'Standard_NC6', 1)]
    #clusters = [('DS12-cpu', 'Standard_DS12_v2',1)]
    configure_computes(ws, clusters)
    gpl_env = Environment.from_conda_specification('gpl_env', 'GPL/environment.yml')
    gpl_run_config = RunConfiguration()
    gpl_run_config.environment = gpl_env

    with open('GPL/gpl/gpl_args.yml', 'rb') as f:
        conf = yaml.safe_load(f.read())  # load the config file

    DATASET_NAME = conf['shared_args']['corpus_name']
    input_data_loc = DataPath(datastore=datastore, path_on_datastore=f'preprocessed_corpus/{DATASET_NAME}.jsonl')
    input_data = Dataset.File.from_files(path=input_data_loc)
    output_folder = 'generated_data'

    gpl_args = [
        '--corpus_path_on_datastore',
        input_data.as_download(output_folder),
        '--path_to_generated_data',
        output_folder,
        '--output_dir',
        'outputs',
        '--generator',
        conf['datagen_args']['generator'],
        '--cross_encoder',
        conf['datagen_args']['cross_encoder'],
        '--batch_size_generation',
        conf['datagen_args']['batch_size_generation'],
        '--queries_per_passage',
        conf['datagen_args']['queries_per_passage'],
        '--negatives_per_query',
        conf['datagen_args']['negatives_per_query'],
        '--gpl_steps',
        conf['shared_args']['gpl_steps'],
        '--batch_size_gpl',
        conf['shared_args']['batch_size_gpl'],
        '--max_seq_length',
        conf['shared_args']['max_seq_length'],
        '--qgen_prefix',
        conf['shared_args']['qgen_prefix'],
    ]

    #step configuration
    generate_gpl_data_step = PythonScriptStep(
        name="generate_gpl_data",
        source_directory='GPL/gpl',
        script_name="training_data.py",
        inputs=[],
        arguments=gpl_args,
        compute_target=clusters[0][0],
        runconfig=gpl_run_config,
        allow_reuse=False
    )

    pipeline_steps = [generate_gpl_data_step]
    preprocess_pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

    print('submitting pipeline ')
    preprocess_pipeline.submit(experiment_name='generate_gpl_data')
