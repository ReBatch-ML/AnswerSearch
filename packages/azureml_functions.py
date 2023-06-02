"""Module that contains reusable functions to interact with azure."""

import os
import yaml
import json
import shutil
from typing import Tuple, List, Dict, Union, Optional
from azureml.core import Workspace, Model, Dataset, Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import ServicePrincipalAuthentication, InteractiveLoginAuthentication
from sentence_transformers import SentenceTransformer, CrossEncoder
from azure.ai.ml import MLClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential


def get_sp_auth():
    """
    Function that returns an authentication object that can be used to authenticate with the azure ml workspace.
    Returns:
        ServicePrincipalAuthentication|InteractiveLoginAuthentication: Authentication object that can be used to authenticate with the azure ml workspace.
    """
    # in case your working on a local machine with the service principle located in the workfolder root
    f_path = '.cloud/.azure/AZURE_SERVICE_PRINCIPAL.json'
    if os.path.exists(f_path):
        with open(f_path) as f:
            cred = f.read()
            os.environ['AZURE_SERVICE_PRINCIPAL'] = cred
    service_principle_str = os.environ.get('AZURE_SERVICE_PRINCIPAL')

    interactive_login = False
    # the sp file needs to exist or the env var is already set, with codespace secrets for example
    if service_principle_str is not None:
        service_principle_cred = json.loads(service_principle_str)
        if service_principle_cred:
            print("Authenticate with environment variable.")
            tenant_id = service_principle_cred["tenant"]
            sp_id = service_principle_cred["appId"]
            sp_pwd = service_principle_cred["password"]
        else:
            if os.path.exists("tenant.txt") and os.path.exists("appid.txt") and os.path.exists("password.txt"):
                print("Authenticate with text files data.")
                tenant_id = open("tenant.txt").read()
                sp_id = open("appid.txt").read()
                sp_pwd = open("password.txt").read()
            else:
                print("Interactive login.")
                interactive_login = True
    else:
        interactive_login = True

    if interactive_login:
        return InteractiveLoginAuthentication(tenant_id="95101651-f23a-4239-a566-84eb874f75f4")
    else:
        sp_auth = ServicePrincipalAuthentication(
            tenant_id=tenant_id, service_principal_id=sp_id, service_principal_password=sp_pwd
        )
        return sp_auth


def get_ws(stage="dev") -> Workspace:
    """Function that returns a workspace for the given stage.
    Args:
        stage (str, optional): One of the deployment staged. Either dev/uat/prod. Defaults to "dev".
    Raises:
        ValueError: In case an invalid stage name is passed.
    Returns:
        Workspace: _description_
    """
    stages = {"dev", "uat", "staging", "prod"}
    if stage not in stages:
        raise ValueError("Invalid stage for workspace: got %s, should be from %s" % (stage, stages))

    sp_auth = get_sp_auth()
    config_path = ".cloud/.azure/{stage}_config.json".format(stage=stage)

    ws = Workspace.from_config(config_path, auth=sp_auth)

    return ws


def get_ml_client(stage: str = "dev"):
    """Function that returns a MLClient for the given stage.

    Args:
        stage (str, optional): Name of the deployment stage. Defaults to "dev".

    Raises:
        ValueError: In case an invalid stage is passed.

    Returns:
        _type_: the mlclient for the given stage that can be used to interact with the ml workspace.
    """
    stages = {"dev", "uat", "staging", "prod"}
    if stage not in stages:
        raise ValueError("Invalid stage for workspace: got %s, should be from %s" % (stage, stages))

    sp_auth = get_sp_auth()
    config_path = ".cloud/.azure/{stage}_config.json".format(stage=stage)

    ml_client = MLClient.from_config(credential=sp_auth, path=config_path)

    return ml_client


def get_secret_client(stage: str = "dev") -> SecretClient:
    """Function that returns a secret client for the given stage.

    Args:
        stage (str, optional): Deployment stage. Defaults to "dev".

    Raises:
        ValueError: In case an invalid stage is passed.

    Returns:
        SecretClient: Secret client for the given stage that can be used to set/get secrets from the keyvault. 
    """
    stages = {"dev", "uat", "staging", "prod"}
    if stage not in stages:
        raise ValueError("Invalid stage for workspace: got %s, should be from %s" % (stage, stages))

    # sp_auth = get_sp_auth()
    # config_path = ".cloud/.azure/{stage}_config.json".format(stage=stage)

    # read vault name from deployment config
    with open(".cloud/.azure/resources_info.json") as f:
        deployment_config = json.load(f)
    vault_name = deployment_config[stage]["keyvault"]

    vault_url = f"https://{vault_name}.vault.azure.net/"
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=vault_url, credential=credential)

    return secret_client


def configure_computes(ws: Workspace, clusters: List[Tuple[str, str, int]]):
    '''
    clusters is a list consisting of the tuples (cluster_name, vm_size, max_nodes)
    e.g. cluster_names = [(cpu-cluster, STANDARD_D2_V2, 2), (gpu-cluster, Standard_NC6, 4)]
    '''
    made_clusters = []
    print("making the clusters:", clusters)
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
        made_clusters.append(cluster)

    return made_clusters


def download_model(workspace: Workspace, model: Dict[str, Union[str, int]], model_type):
    """
    Function downloads the model and copies it to the models/model_type folder
    :param model: Dictionary that contains the name and version of the model that needs to be downloaded
    """
    print(f'download bi_encoder{model["name"]}:{model["version"]}')
    model_path = Model.get_model_path(model_name=model['name'], version=model['version'], _workspace=workspace)
    shutil.copytree(src=model_path, dst=f"models/{model_type}")

    return model_path


def combine_models(
    config_file: str = "configs/deployment_config.yaml",
    language_config: str = "configs/model_languages.yaml",
    bi_encoder: Tuple[str, int] = None,
):
    """
    Combines 2 models that are on the model registry into 1 model and registers it again, so it can be used for inference
    :config_file: Location to a config yaml file that contains info about the deployment
    :language_config: Location to a config yaml file that contains info about what model to use for which language
    :param bi_encoder: (model_name, model_version) as stated in the model registry for the first model if empty the standard untrained model will be used
    """
    ws = get_ws("dev")

    with open(config_file, 'r') as file:
        config = yaml.safe_load(stream=file)

    with open(language_config, 'r') as file:
        language_models = yaml.safe_load(stream=file)

    language = config['corpus_language']

    bi = language_models[language.lower()]['bi_encoder']
    cross = language_models[language.lower()]['cross_encoder']

    if bi_encoder:
        registry_model = {"name": bi_encoder[0], "version": bi_encoder[1]}
        _ = download_model(ws, registry_model, model_type="bi_encoder")
    else:
        bi_model = SentenceTransformer(bi)
        bi_model.save("models/bi_encoder")

    model = CrossEncoder(cross)
    model.save("models/cross_encoder")

    Model.register(
        ws,
        model_path="models",
        model_name="bi_cross_encoders",
        description="Combination of a bi- and cross-encoder that is needed to do inference"
    )
    shutil.rmtree('models')


def upload_folder_to_datastore(path_on_datastore, local_data_folder, stage='dev'):
    """Function that will upload a local folder to the default datastore of the dev environment

    Args:
        path_on_datastore (string): Path on datastore where the folder is uploaded to
        local_data_folder (string): Path to the local folder that needs to be uploaded
        stage (string, optional): Name of the environment stage that the data needs to be uploaded to
    """

    workspace = get_ws(stage)
    # Gets the default datastore, this is where we are going to save our new data
    datastore = workspace.get_default_datastore()

    # Under which path do we want to save our new data
    datastore_path = path_on_datastore  #/{}".format(str_date_time)

    # Select the directory where we put our processed data and upload it to our datastore
    preprocessed = Dataset.File.upload_directory(local_data_folder, (datastore, datastore_path), overwrite=True)
