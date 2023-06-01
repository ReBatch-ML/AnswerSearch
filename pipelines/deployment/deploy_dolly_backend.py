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
from utils import create_endpoint


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

    endpoint_name = "dolly-endpoint"
    print(f"Endpoint name: {endpoint_name}")
    existing_endpoints = [e.name for e in ml_client.online_endpoints.list()]
    print(f"Existing endpoints: {existing_endpoints}")

    # lower because it seems that the endpoint names are lowercased and case insensitive, so compare lowercased names
    compare = [endpoint_name.lower() != e.lower() for e in existing_endpoints]
    # no endpoint has the same name, so we can create a new one
    if all(compare):
        create_endpoint(ml_client=ml_client, endpoint_name=endpoint_name, description="Endpoint that will perform dolly summarization on the flan answers.")

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
