# README
## Deploy endpoint
In case the dataset needed is too big or redeployment occurs often, it is best to deploy the endpoint in a pre-created docker container with the corpus embeddings already downloaded into.

In the environments folder are:
1. _Dockerfile_ 
2. _download_ds.py_ script 
3. _online_scoring_environemnt.yml_ 

### The Dockerfile:

    FROM azureml-image

    Installation of conda

    COPY environment.yml
    RUN install the conda env with the yml specs

    COPY download.py script
    RUN download.py script

The image from which to start should always be a precurated image from azureml (https://github.com/Azure/AzureML-Containers). This way any necessary packages needed for online inference are included in the environment. 

### download.py script
In this script you have to specify the locations of the 2 datasets. This can be done with the blobstore name and a relative path on the blob store. That path should point towards the directory where the dataset and the faiss index are stored.

### Build the image
Can be done locally, even more so, it is advised. Using docker inside a docker container is potentially a bit weird anyway.

    cd environments
    az login --tenant rebatch.onmicrosoft.com
    acr_url=acrsemanticsearch.azurecr.io # replace if different
    az acr login -n $acr_url

    local_image_name=semantic_search_fullpublic_fullstaffreg # can be replaced
    docker build -t $local_image_name .

    image_name=$local_image_name # does not have to be the same
    acr_image_name=$acr_url/$image_name

    docker tag $local_image_name $acr_image_name

    docker push $acr_image_name



## Turn off deployments to safe money

### online managed deployments:
Simply delete the deployment from the UI in the machine learning studio. The endpoint itself can remain.

1. Open the ML studio
2. Navigate to Endpoints in the sidebar
3. Click on the endpoint belonging to the deployment
4. Click on the thrashcan icon on the top right of the deployment pane

### frontend UI:
1. Go to the resource group where the website resources reside:
    1. App Service
    2. App Service plan
2. Delete both resources


# Data Science Lifecycle Base Repo

Use this repo as a template repository for data science projects using the Data Science Life Cycle Process. This repo is meant to serve as a launch off point. Our goal is to introduce only **minimum viable opinions** into the structure of this repo in order to make this repository/framework useful across a variety of data science projects and workflows. Therefore, we will tend to err to the side of omitting something if we're not confident that it's widely useful or think it's overly opinionated. That shouldn't stop you from forking this repo and adapting it to fit the needs of your project/team/organization.

With that in mind, if there is something that you think we're missing or should change, open an issue and we'll talk!

## Get started.

The only manual step required is that you have to manually create the labels. The label names, descriptions, and color codes can be found in the [.github/labels.yaml](/.github/labels.yaml) file. For more information on creating labels, review the GitHub docs [here](https://help.github.com/en/github/managing-your-work-on-github/creating-a-label).

## Contributing

Issues and suggestions for this template repo should be opened in the main [dslp repo](https://github.com/MicrosoftDSST/dslp/issues).

## Default Directory Structure

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
├── pipelines           # for pipeline orchestrators i.e. AzureML Pipelines, Airflow, Luigi, etc.
├── src
│   ├── datasets        # code for creating or getting datasets
│   ├── deployment      # code for deploying models
│   ├── features        # code for creating features
│   └── models          # code for building and training models
├── setup.py            # if using python, for finding all the packages inside of code.
└── tests               # for testing your code, data, and outputs
    ├── data_validation
    └── unit
```