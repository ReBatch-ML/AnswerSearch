""" Small test to check the online endpoint """
import urllib.request
import json
import os
import ssl
from time import time


def allowSelfSignedHttps(allowed):
    """_summary_

    Args:
        allowed (_type_): _description_
    """
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data = {"query": ["What is the expected rise in unemployment rate for 2021?"]}

body = str.encode(json.dumps(data))

stream = os.popen(
    'az ml online-endpoint get-credentials -n online-endpoint -o tsv --query primaryKey -g Semantic_Search -w SemanticSearch_TRAIN'
)
api_key = stream.read().replace('\n', '')

url = 'https://online-endpoint.westeurope.inference.ml.azure.com/score'

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {
    'Content-Type': 'application/json',
    'Authorization': ('Bearer ' + api_key),
    'azureml-model-deployment': 'standard-f8s-v2'
}

req = urllib.request.Request(url, body, headers)

try:
    print("Send request...")
    start = time()
    response = urllib.request.urlopen(req)
    inter = time()
    result = response.read().decode('utf8')
    result = json.loads(result)
    end = time()

    with open("given_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f)

    print(f"Total time is {end-start}s of which {inter-start}s are used to access the endpoint.")
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
