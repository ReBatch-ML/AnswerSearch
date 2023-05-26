"""Script to deploy the backend for the dolly summarization on the flan answers."""
from packages.azureml_functions import get_ml_client, get_secret_client
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment, CodeConfiguration, OnlineRequestSettings, ProbeSettings
from azure.identity import AzureCliCredential
import argparse

from azure.mgmt.authorization import AuthorizationManagementClient
import uuid
from azure.mgmt.authorization.v2020_10_01_preview.models import (
    RoleAssignment,
    RoleAssignmentCreateParameters,
)


def create_endpoint(ml_client, endpoint_name):
    """Initialize an endpoint dolly in azure

    Args:
        ml_client (MLClient): the mlclient that can be used to interact with the ml workspace.
        endpoint_name (String): name for the endpoint
    """
    # Create the endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name, description="Endpoint that will perform dolly summarization on the flan answers."
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


def create_deployment(ml_client, endpoint_name):
    """Initialize a deployment for the dolly backend"""

    env = Environment(
        conda_file="environments/dolly_backend_environment.yml",
        image="mcr.microsoft.com/azureml/curated/acpt-pytorch-1.13-py38-cuda11.7-gpu:1",
    )

    onlineRequestSettings = OnlineRequestSettings(request_timeout_ms=90000)
    livenessProbeSettings = ProbeSettings(failure_threshold=30, initial_delay=2000, period=100, success_threshold=1)

    dolly = ml_client.models.get(name="dolly", version='1')

    deployment = ManagedOnlineDeployment(
        name="dolly",
        endpoint_name=endpoint_name,
        model=dolly,
        environment=env,
        code_configuration=CodeConfiguration(code="src/deployment/backend", scoring_script="dolly_score.py"),
        instance_type="Standard_NC24ads_A100_v4",
        liveness_probe=livenessProbeSettings,
        instance_count=1,
        request_settings=onlineRequestSettings
    )

    deployment = ml_client.online_deployments.begin_create_or_update(deployment)
    return deployment


def main():
    """Create endpoint and deployment for the backend; store url and key in keyvault
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default="dev")
    args = parser.parse_args()

    ml_client = get_ml_client(stage=args.stage)

    endpoint_name = "GSC-dolly"
    print(f"Endpoint name: {endpoint_name}")
    existing_endpoints = [e.name for e in ml_client.online_endpoints.list()]
    print(f"Existing endpoints: {existing_endpoints}")

    # lower because it seems that the endpoint names are lowercased and case insensitive, so compare lowercased names
    compare = [endpoint_name.lower() != e.lower() for e in existing_endpoints]
    # no endpoint has the same name, so we can create a new one
    if all(compare):
        create_endpoint(ml_client=ml_client, endpoint_name=endpoint_name)

    create_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint_name,
    )

    endpoint_url = ml_client.online_endpoints.get(endpoint_name).scoring_uri
    endpoint_key = ml_client.online_endpoints.get_keys(name=endpoint_name).primary_key

    secret_client = get_secret_client(stage=args.stage)

    secret_client.set_secret(name="dolly-url", value=endpoint_url)
    secret_client.set_secret(name="dolly-key", value=endpoint_key)


if __name__ == "__main__":
    main()
