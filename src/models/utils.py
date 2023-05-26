"""Utility functions for models that can be used inside azureml experiments. So they need to be able to run without extra info found in the repository."""
from azureml.core import Run


def register_model(
    run: Run, model_name: str, model_location: str, epoch: int, tags=None, properties=None, description=None
):
    """
    Register a model in an AzureML workspace, during an experiment run.
    Args:
        run (Run): Run object for the experiment.
        model_name (str): name under which to register the model.
        model_location (str): location where the model is saved locally.
        epoch (int): epoch number.
        tags (Dict, optional): tags registered alongside the model. Defaults to None.
        properties (Dict, optional): properties registered alongside the model. Defaults to None.
        description (str, optional): text that describes the model. Defaults to None.
    """
    if "." in model_location:
        model_extension = f".{model_location.rsplit('.', 1)[-1]}"
    else:
        model_extension = ""

    upload_path = f"outputs/epoch_{epoch}/model{model_extension}"
    # run.upload_file(name=upload_path, path_or_stream=model_location)
    run.upload_folder(name=upload_path, path=model_location)
    run.register_model(
        model_name=model_name, model_path=upload_path, tags=tags, properties=properties, description=description
    )
