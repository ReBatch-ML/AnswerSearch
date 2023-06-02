"""script to deploy the milvus kubernetes cluster"""
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient


def start_milvus():
    """start the milvus kubernetes cluster
    """
    credential = DefaultAzureCredential()
    subscription_id = "9705804d-0f2b-43de-a254-038eaf5c9255"
    resource_group_name = "AnswerSearch"
    cluster_name = "Milvus-kube"

    client = ContainerServiceClient(credential, subscription_id)
    client.managed_clusters.begin_start(resource_group_name, cluster_name)

def stop_milvus():
    """stop the milvus kubernetes cluster
    """
    credential = DefaultAzureCredential()
    subscription_id = "9705804d-0f2b-43de-a254-038eaf5c9255"
    resource_group_name = "AnswerSearch"
    cluster_name = "Milvus-kube"

    client = ContainerServiceClient(credential, subscription_id)
    client.managed_clusters.begin_stop(resource_group_name, cluster_name)


if __name__ == "__main__":
    start_milvus()
