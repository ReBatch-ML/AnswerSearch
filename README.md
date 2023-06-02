# README


## Deploy backend
### online managed deployments:
Run script:
'''
python pipelines/deployment/deploy_all_backend.py start
'''

This will start the Milvus kubernetes service, check if the milvus server is online and then deploy the semsearch, flan and dolly backend modules.

## Turn off deployments to safe money

### online managed deployments:
Run script:
'''
python pipelines/deployment/deploy_all_backend.py stop
'''

This will stop the milvus kubernetes service and delete the semsearch, flan and dolly backend endpoints.

### frontend UI:
1. Go to the resource group where the website resources reside:
    1. App Service
    2. App Service plan
2. Delete both resources


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
