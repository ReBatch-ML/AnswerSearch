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
    input_data_loc = DataPath(datastore=datastore, path_on_datastore=f'gpl_training_data/{DATASET_NAME}/train')
    input_data_train = Dataset.File.from_files(path=input_data_loc)
    input_data_loc = DataPath(datastore=datastore, path_on_datastore=f'gpl_training_data/{DATASET_NAME}/test')
    input_data_test = Dataset.File.from_files(path=input_data_loc)

    train_data_dir = 'generated_data'
    test_data_dir = 'test_data'
    gpl_args = [
        '--path_to_generated_data', train_data_dir, '--output_dir', 'outputs', '--evaluation_data', test_data_dir,
        '--base_ckpt', conf['train_args']['base_ckpt_bi_encoder'], '--gpl_steps', conf['shared_args']['gpl_steps'],
        '--batch_size_gpl', conf['shared_args']['batch_size_gpl'], '--max_seq_length',
        conf['shared_args']['max_seq_length'], '--qgen_prefix', conf['shared_args']['qgen_prefix']
    ]

    #step configuration
    gpl_train_step = PythonScriptStep(
        name="gpl_train",
        source_directory='GPL/gpl',
        script_name="train.py",
        inputs=[input_data_train.as_download(train_data_dir),
                input_data_test.as_download(test_data_dir)],
        arguments=gpl_args,
        compute_target=clusters[0][0],
        runconfig=gpl_run_config,
        allow_reuse=False
    )

    pipeline_steps = [gpl_train_step]
    training_pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

    print('submitting pipeline ')
    training_pipeline.submit(experiment_name='gpl_train')
