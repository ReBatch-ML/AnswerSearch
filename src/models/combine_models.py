""" Combine bi and cross encoder into 1 registered model """
import yaml
import shutil
from azureml.core import Model
from packages.azureml_functions import get_ws

ws = get_ws()

with open("src/deployment/deployment.yaml", 'r') as file:
    config = yaml.safe_load(stream=file)

bi_encoder_config = config['models'][0]
cross_encoder_config = config['models'][1]
print(f'download bi_encoder{bi_encoder_config["name"]}:{bi_encoder_config["version"]}')
bi_encoder_path = Model.get_model_path(
    model_name=bi_encoder_config['name'], version=bi_encoder_config['version'], _workspace=ws
)
print(f'download cross_encoder{cross_encoder_config["name"]}:{cross_encoder_config["version"]}')
cross_encoder_path = Model.get_model_path(
    model_name=cross_encoder_config['name'], version=cross_encoder_config['version'], _workspace=ws
)

shutil.copytree(src=bi_encoder_path, dst="models/bi_encoder")
shutil.copytree(src=cross_encoder_path, dst="models/cross_encoder")

Model.register(
    ws,
    model_path="models",
    model_name="bi_cross_encoders",
    description="Combination of a bi- and cross-encoder that is needed to do inference"
)
shutil.rmtree('models')
