"""Deploy the backend to Azure ML by reading the specs in the config file"""
from packages.azureml_functions import get_ml_client, get_secret_client
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment, CodeConfiguration, OnlineRequestSettings
from azure.identity import AzureCliCredential
import argparse
import json
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.authorization.v2018_01_01_preview.models import RoleDefinition
import uuid
from azure.mgmt.authorization.v2020_10_01_preview.models import (
    RoleAssignment,
    RoleAssignmentCreateParameters,
)
from azureml.core import Environment as Environmentv1
from azureml.core import Workspace
import yaml
import time
from azure.ai.ml import MLClient
from pathlib import Path


def create_endpoint(ml_client: MLClient, endpoint_name: str):
    """
    Creates an endpoint to the Azure ML workspace belonging to the MLClient that has access to the storage account
    Args:
        ml_client (MLClient): ml_client that can access the Azure ML workspace
        endpoint_name (str): name for the endpoint
    """
    # Create the endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name, description="Endpoint that will perform similaritysearch for an image."
    )
    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    poller.result()

    # 2 extra clients needed for creating a new role assignment
    credential = AzureCliCredential()

    role_definition_client = AuthorizationManagementClient(
        credential=credential,
        subscription_id=ml_client.subscription_id,
        api_version="2018-01-01-preview",
    )

    role_assignment_client = AuthorizationManagementClient(
        credential=credential,
        subscription_id=ml_client.subscription_id,
        api_version="2020-10-01-preview",
    )

    # retrieve the system assigned identity of the endpoint so we can assing the role to it
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    system_principal_id = endpoint.identity.principal_id

    # create the role for reading data from blob storage
    role_name = "Storage Blob Data Reader"
    scope = ml_client.datastores.get_default().id

    role_defs = role_definition_client.role_definitions.list(scope=scope)
    role_def = next((r for r in role_defs if r.role_name == role_name))

    # assign the role
    role_assignment_client.role_assignments.create(
        scope=scope,
        role_assignment_name=str(uuid.uuid4()),
        parameters=RoleAssignmentCreateParameters(role_definition_id=role_def.id, principal_id=system_principal_id),
    )


def create_deployment(ml_client: MLClient, endpoint_name: str, env_vars=None) -> ManagedOnlineDeployment:
    """
    Creates a deployment to the Azure ML workspace belonging to the MLClient.

    Args:
        ml_client (MLClient): ml client that can access the Azure ML workspace
        endpoint_name (str): name for the deployment
        env_vars (Dict, optional): Dictionary that has all env vars needed for the deployment scoring script to work. Defaults to None.

    Returns:
        ManagedOnlineDeployment: The online deployment that was created
    """
    embeddings_model = ml_client.models.get(
        name=env_vars["BI_CROSS_ENCODER_NAME"], version=env_vars["BI_CROSS_ENCODER_VERSION"]
    )

    env = Environment(
        conda_file="environments/backend_environment.yml",
        image="mcr.microsoft.com/azureml/curated/acpt-pytorch-1.13-py38-cuda11.7-gpu:1",
    )

    onlineRequestSettings = OnlineRequestSettings(request_timeout_ms=15000)

    deployment = ManagedOnlineDeployment(
        name="semantic-search",
        endpoint_name=endpoint_name,
        model=embeddings_model,
        environment=env,
        code_configuration=CodeConfiguration(code="src/deployment/backend", scoring_script="score.py"),
        instance_type="Standard_DS4_v2",
        instance_count=1,
        environment_variables=env_vars,
        request_settings=onlineRequestSettings
    )

    deployment = ml_client.online_deployments.begin_create_or_update(deployment)
    return deployment


def set_env_vars(ml_client: MLClient, deployment_config, config_file: str):
    """
    Sets the environment variables for the deployment
    Args:
        ml_client (MLClient): MLClient that can access the Azure ML workspace
        deployment_config (Dict): Dictionary that has the info from the deployment_config.yaml file
        config_file (str): Path to the config file in the .azure folder that has azure resource information

    Returns:
        Dict: Dictionary that has all the environment variables needed for the deployment scoring script to work
    """
    configs = json.load(open(config_file, 'r'))

    default_ds = ml_client.datastores.get_default()
    configs["STORAGE_ACCOUNT_NAME"] = default_ds.account_name
    configs["STORAGE_CONTAINER_NAME"] = default_ds.container_name
    configs["FOLDERS"] = ",".join(
        [str(Path(deployment_config["root"], corpus["location"])) for corpus in deployment_config["corpora"]]
    )
    configs["CORPORA"] = ",".join([corpus["name"] for corpus in deployment_config["corpora"]])
    configs["BI_CROSS_ENCODER_NAME"] = deployment_config["bi_cross_encoder_name"]
    configs["BI_CROSS_ENCODER_VERSION"] = deployment_config["bi_cross_encoder_version"]

    return configs


def main():
    """
    Main function that creates the endpoint and deployment and saves the api key and url to the keyvault of the deployment stage.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default="dev")
    args = parser.parse_args()

    path_to_config = f".cloud/.azure/{args.stage}_config.json"
    ml_client = get_ml_client(stage=args.stage)

    with open("configs/deployment_config.yaml", 'r') as file:
        deployment_config = yaml.safe_load(stream=file)

    client_name = deployment_config["client_name"]
    endpoint_name = f"{client_name}-semantic-search"
    print(f"Endpoint name: {endpoint_name}")
    existing_endpoints = [e.name for e in ml_client.online_endpoints.list()]
    print(f"Existing endpoints: {existing_endpoints}")

    # lower because it seems that the endpoint names are lowercased and case insensitive, so compare lowercased names
    compare = [endpoint_name.lower() != e.lower() for e in existing_endpoints]
    # no endpoint has the same name, so we can create a new one
    if all(compare):
        create_endpoint(ml_client=ml_client, endpoint_name=endpoint_name)

    env_vars = set_env_vars(ml_client=ml_client, deployment_config=deployment_config, config_file=path_to_config)
    create_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint_name,
        env_vars=env_vars,
    )

    endpoint_url = ml_client.online_endpoints.get(endpoint_name).scoring_uri
    endpoint_key = ml_client.online_endpoints.get_keys(name=endpoint_name).primary_key

    secret_client = get_secret_client(stage=args.stage)

    secret_client.set_secret(name="api-url", value=endpoint_url)
    secret_client.set_secret(name="api-key", value=endpoint_key)


if __name__ == "__main__":
    main()
