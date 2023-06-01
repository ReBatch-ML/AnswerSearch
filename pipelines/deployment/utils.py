"""Utility functions for deployment."""
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.identity import AzureCliCredential
from azure.mgmt.authorization import AuthorizationManagementClient
import uuid
from azure.mgmt.authorization.v2020_10_01_preview.models import (
    RoleAssignment,
    RoleAssignmentCreateParameters,
)


def create_endpoint(ml_client, endpoint_name, description):
    """
    Function for creating an endpoint if it does not exist yet.

    Args:
        ml_client (MLClient): connection to Azure ML workspace
        endpoint_name (str): name of the endpoint to create
    """
    existing_endpoints = [e.name for e in ml_client.online_endpoints.list()]
    print("Existing endpoints: ", existing_endpoints)
    # lower because it seems that the endpoint names are lowercased and case insensitive, so compare lowercased names
    compare = [endpoint_name.lower() != e.lower() for e in existing_endpoints]
    # no endpoint has the same name, so we can create a new one
    if not all(compare):
        return

    # Create the endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name, description=description
    )
    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    poller.result()
    """
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
    """
