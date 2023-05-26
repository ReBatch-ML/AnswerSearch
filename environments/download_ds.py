"""Script to download the datasets from Azure Blob Storage."""
import os
from azure.storage.blob import BlobServiceClient

with open("connection_string.txt") as f:
    connection_string = f.readline()

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blobstore = "azureml-blobstore-04f263a3-f5c3-4204-9686-0f886c746b64"


def download_blobs(blob_store, source_loc, dest_loc):
    """
    Download the files from Azure Blob Storage to local storage.

    Args:
        blob_store (str): Name of the blob store.
        source_loc (str): Location on the blob store.
        dest_loc (str): Location where to write to.
    """
    container_client = blob_service_client.get_container_client(blob_store)
    blob_gen = container_client.list_blobs(name_starts_with=source_loc)

    for b in blob_gen:
        # step 1: Download the file to local storage
        file_name = b.name.rsplit('/', 1)[-1]
        print("Download:", file_name)

        os.makedirs(dest_loc, exist_ok=True)
        target_path = os.path.join(dest_loc, file_name)
        with open(target_path, 'wb') as f:
            f.write(container_client.download_blob(b.name, timeout=7200).readall())


print("Start downloading the public registry...")
public_reg_loc = "corpus_ds_with_embedding/corpus_all_length_250_multi-qa-mpnet-base-dot-v1_TRAINED_2022-10-21-14-33-13_version_45/"
download_blobs(blobstore, public_reg_loc, "/corpora/public_registry")

print("Start downloading the staff regulations of EU officials...")
staff_reg_eu_loc = "corpus_ds_with_embedding/stafreg_eu_officials_multi-qa-mpnet-base-dot-v1_TRAINED_2022-10-21-14-33-13_version_45/"
download_blobs(blobstore, staff_reg_eu_loc, "/corpora/stafreg_registry_officials")

print("Start downloading the staff regulations of other servants...")
staff_reg_other_loc = "corpus_ds_with_embedding/stafreg_other_servants_multi-qa-mpnet-base-dot-v1_TRAINED_2022-10-21-14-33-13_version_45/"
download_blobs(blobstore, staff_reg_other_loc, "/corpora/stafreg_registry_others")
