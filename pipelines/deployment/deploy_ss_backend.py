"""Script to deploy the semantic search backend to Azure ML."""
from packages.azureml_functions import get_ml_client, get_secret_client
from azure.ai.ml.entities import ManagedOnlineDeployment, Environment, CodeConfiguration, OnlineRequestSettings
import argparse
from utils import create_endpoint


def create_deployment(ml_client, endpoint_name, env_vars=None):
    """
    Create a deployment for the semantic search backend.

    Args:
        ml_client (MLClient): Connection to Azure ML workspace.
        endpoint_name (str): name of the endpoint to deploy to.
        env_vars (Dict, optional): Environment variables that should be set in the backend. Defaults to None.

    Returns:
        ManagedOnlineDeployment: the created deployment
    """

    embeddings_model = ml_client.models.get(name="bi_cross_encoders", version='1')

    env = Environment(
        # conda_file="environments/backend_environment.yml",
        image="9a69c7869338423eaae7eb32fb63f9db.azurecr.io/semantic_search_fullpublic_fullstaffreg:latest",
    )

    onlineRequestSettings = OnlineRequestSettings(request_timeout_ms=20000)

    deployment = ManagedOnlineDeployment(
        name="semantic-search",
        endpoint_name=endpoint_name,
        model=embeddings_model,
        environment=env,
        code_configuration=CodeConfiguration(code="src/deployment/backend", scoring_script="semsearch_score.py"),
        instance_type="Standard_NC4as_T4_v3",
        instance_count=1,
        environment_variables=env_vars,
        request_settings=onlineRequestSettings
    )

    deployment = ml_client.online_deployments.begin_create_or_update(deployment)
    return deployment


def main():
    """
    Main function for the deployment script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default="dev")
    args = parser.parse_args()

    ml_client = get_ml_client(stage=args.stage)

    endpoint_name = f"semantic-search"
    print(f"Endpoint name: {endpoint_name}")

    create_endpoint(ml_client=ml_client, endpoint_name=endpoint_name)

    create_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint_name,
        env_vars=None,
    )

    endpoint_url = ml_client.online_endpoints.get(endpoint_name).scoring_uri
    endpoint_key = ml_client.online_endpoints.get_keys(name=endpoint_name).primary_key

    secret_client = get_secret_client(stage=args.stage)

    secret_client.set_secret(name="semsearch-url", value=endpoint_url)
    secret_client.set_secret(name="semsearch-key", value=endpoint_key)


if __name__ == "__main__":
    main()
