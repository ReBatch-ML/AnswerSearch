# README


## Deploy backend
### Online managed deployments:
Run script in source directory:
```
python pipelines/deployment/deploy_all_backend.py start
```

This will start the Milvus kubernetes service, check if the milvus server is online and then deploy the semsearch, flan and dolly backend modules.
### Frontend UI:
To launch the webapp, run script in source directory:
```
./pipelines/deployment/deploy_webapp.sh
```
If you want to run the app locally, first go to the backend endpoints in azure ML studio of the AnswerSearch resourcegroup. If the deployment was successful you can find the api url and keys under "Consume". Copy the urls and keys for all 3 backend deployments and put them in a json file as specified in pipelines/deployment/deploy_localapp.sh.
Once this is done run script in source directory:
```
./pipelines/deployment/deploy_localapp.sh
```

## Turn off deployments to safe money

### Online managed deployments:
Run scriptin source directory:
```
python pipelines/deployment/deploy_all_backend.py stop
```

This will stop the milvus kubernetes service and delete the semsearch, flan and dolly backend endpoints.

### Frontend UI:
1. Go to the AnswerSearch resource group where the website resources reside:
    1. App Service
    2. App Service plan
2. Delete both resources

## Check on Milvus Kubernetes service
### In Azure Portal (installation)
Acces Kubernetes service in Azure Portal:
1. Go to the AnswerSearch resource group where the Kubernetes resources reside
2. Go to "Milvus-kube"
3. Click on "Connect"
4. Click on "Open Cloud Shell"
To view external ip of the Milvus service to connect to, run:
```
Kubectl get services
```
To view status of the milvus services, run:
```
Kubectl get pods
```
All 3 need to be running (1/1) in order for Milvus to work.

In order to install Milvus run:
```
helm install my-release milvus/milvus --set service.type=LoadBalancer --set cluster.enabled=false --set etcd.replicaCount=1 --set minio.mode=standalone --set pulsar.enabled=false
```
To uninstall Milvus run:
```
helm uninstall my-release
```
(unistalling Milvus will delete all collections)
### With Python SDK (Managing Collections)
The file packages/milvus_functions.py contains fucntions to:
1. Connect to Milvus
2. Create collection
3. Change index
4. Insert data from corpus
5. Perform search on vectors
6. Calculate recall using test file for GSC corpus
7. Drop collection

## Directory Structure

```
├── .cloud              # for storing cloud configuration files and templates (e.g. ARM, Terraform, etc)
├── .github
│   ├── ISSUE_TEMPLATE
│   │   ├── Ask.md
│   │   ├── Data.Aquisition.md
│   │   ├── Data.Create.md
│   │   ├── Experiment.md
│   │   ├── Explore.md
│   │   └── Model.md
│   ├── labels.yaml
│   └── workflows
├── .gitignore
├── README.md
├── data                # directory is for consistent data placement. contents are gitignored by default.
│   ├── README.md
│   ├── interim         # storing intermediate results (mostly for debugging)
│   ├── processed       # storing transformed data used for reporting, modeling, etc
│   └── raw             # storing raw data to use as inputs to rest of pipeline
├── docs
│   ├── code            # documenting everything in the code directory (could be sphinx project for example)
│   ├── data            # documenting datasets, data profiles, behaviors, column definitions, etc
│   ├── media           # storing images, videos, etc, needed for docs.
│   ├── references      # for collecting and documenting external resources relevant to the project
│   └── solution_architecture.md    # describe and diagram solution design and architecture
├── environments
├── notebooks
├── packages            # contains utility functions
├── pipelines           # for pipeline orchestrators i.e. AzureML Pipelines, Airflow, Luigi, etc.
├── src
│   ├── datasets        # code for creating or getting datasets
│   ├── deployment      # code for deploying models
│   ├── features        # code for creating features
│   └── models          # code for building and training models
├── setup.py            # if using python, for finding all the packages inside of code.

```
