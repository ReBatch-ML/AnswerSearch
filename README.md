# Semantic Search repository
Docstrings are expected in all modules, classes and functions (except __init__.py). The format of the docstring is expected to be the same as google's.
There is a extension in the devcontainer that can generate docstrings with the google style.


## Creating a new project:

When creating a repository from this template ensure that the repo name starts with a short client name and then a suffix. But ideally remain consistent by taking the first suffix from the list. *ex: CepaSemanticSearch*

Suffixes:
1. SemanticSearch
2. semanticSearch
3. Semanticsearch
4. -semantic-search
5. _semantic_search
6. semsearch Semsearch
7. SemSearch

# Workflows:
## 1: Setup the Semantic Search project
This workflow will create the azure resources necessary for the semantic search projects. It will create a resource group with multiple ML studios in it depending on what deployment stages are specified in the config file. (For now default is 'dev' and 'uat'). 

Stage names should be dev,uat or prod to be compatible with some functions in the azureml_functions package where we check if a stage is either of those names.

---
## 2: Setup pretrained inference demo
The config file at configs/model_languages.yaml is very important here. It configures how the inference demo will have to be run. The language of the corpus, where the corpus is located and what their names are, are all specified in this file. 

It controls the creation of the embeddings, the backend and the frontend of the demo. More explanation is found inside the config file itself


Once the script in src/datasets/preprocess.py is adapted to the specific corpus and ensured to work with the script at src/models/embedding_creator.py and the config file is correct, the workflow can be manually triggered. 

In this workflow, the preprocessing is triggered followed by the creation of embeddings. Once this is done, both the backend and frontend will be deployed.

---
## 3: How to manually trigger workflows

![How to run manual worklows](docs/media/run_workflow.jpg?raw=true "How to run workflow manually")

1. Go to the actions tab in GitHub
2. Select which workflow you'd like to trigger
    > only workflows that are triggered on workflow_dispatch can be manually triggered
3. Click on Run workflow and chose from which branch you'd want to run the workflow

# Preprocessing
Ideally preprocessing will accept a folder per corpus. Each folder can have one or more files that make up the corpus. These folders should be located in the 'raw_corpus_path' in the config (which can be set depending on where you upload the raw corpus to).

For each folder/corpus the script should return a preprocessed jsonl file. These jsonl files have a preprocessed text field and some metadata like a unique ID, title, etc. This result will be uploaded to the processed/ folder with subfolders being specified by the config file corpora locations.

# Embedding creation
This script will take the processed folder and create embeddings. It will create a huggingface dataset for each corpus. Metadata present in the jsonl file from the preprocessing will need to be added to the dataset aswell. That should probably be the majority of the changes that need to be done to the script. 

The embedding datasets will be outputted to the place specified in the config file on root/corpora.location path. This path will both function as the output location for the embedding pipeline as well as input for the demo backend to find the corpora. 

` `  
` `  
` `  
` `  
` `  
` `  
` `  

# Data Science Lifecycle Base Repo

Use this repo as a template repository for data science projects using the Data Science Life Cycle Process. This repo is meant to serve as a launch off point. Our goal is to introduce only **minimum viable opinions** into the structure of this repo in order to make this repository/framework useful across a variety of data science projects and workflows. Therefore, we will tend to err to the side of omitting something if we're not confident that it's widely useful or think it's overly opinionated. That shouldn't stop you from forking this repo and adapting it to fit the needs of your project/team/organization.

With that in mind, if there is something that you think we're missing or should change, open an issue and we'll talk!

## Get started.

The only manual step required is that you have to manually create the labels. The label names, descriptions, and color codes can be found in the [.github/labels.yaml](/.github/labels.yaml) file. For more information on creating labels, review the GitHub docs [here](https://help.github.com/en/github/managing-your-work-on-github/creating-a-label).

## Contributing

Issues and suggestions for this template repo should be opened in the main [dslp repo](https://github.com/MicrosoftDSST/dslp/issues).

## Default Directory Structure

```
├── .cloud              # for storing cloud configuration files
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
├── configs             
│   ├── deployment_config.yaml    # configuration file that has all configurable parameters to make a deployment
│   ├── model_languages.yaml      # Config that defines which base models to use for which language
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
├── packages            # folder where reusable code is placed such as functions that interact with azure etc.
├── src
│   ├── datasets        # code for creating or getting datasets (preprocessing)
│   ├── deployment      # code for deploying models (frontend and backend)
│   ├── features        # code for creating features
│   └── models          # code for building and training models
├── setup.py            # if using python, for finding all the packages inside of code.
└── tests               # for testing your code, data, and outputs
    ├── data_validation
    └── unit
```