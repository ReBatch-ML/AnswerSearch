// Parameters
@minLength(2)
@maxLength(10)
@description('Prefix for all resource names.')
param clientName string

@minLength(1)
@maxLength(3)
@description('Which stages in the deployment to create out of dev,uat,prod')
param deployment_stages array = ['dev']

@description('Azure region used for the deployment of all resources.')
param location string = 'westeurope'

@description('Set of tags to apply to all resources.')
param tags object = {}

// @secure()
// @description('Service principal json information')
// param service_principal string = ''

// Variables
var name = toLower('${clientName}')
var stages = deployment_stages


// Create a short, unique suffix, that will be unique to each resource group
var uniqueSuffix = substring(uniqueString(resourceGroup().id), 0, 4)
var keyvault_names = [for deployment_stage in stages: 'kv-${name}-${deployment_stage}-${uniqueSuffix}']
var storage_names = [for deployment_stage in stages: 'storagess${name}${deployment_stage}${uniqueSuffix}']
var container_registry_name = 'cr${name}${uniqueSuffix}'
var workspace_names = [for deployment_stage in stages: 'workspace-${name}-${deployment_stage}']



module keyvault 'modules/keyvault.bicep' = [for (deployment_stage, i) in stages: {
  name: 'kv-${name}-${uniqueSuffix}-deployment-${deployment_stage}'
  params: {
    location: location
    keyvaultName: keyvault_names[i]
    tags: tags
    // sp_secret:service_principal
  }
}]

module storage 'modules/storage.bicep' = [for (deployment_stage, i) in stages: {
  name: 'st-semsearch${name}${uniqueSuffix}-deployment-${deployment_stage}'
  params: {
    location: location
    storageName: storage_names[i]
    storageSkuName: 'Standard_LRS'
    tags: tags
  }
}]

module containerRegistry 'modules/containerregistry.bicep' = {
  name: 'cr${name}${uniqueSuffix}-deployment'
  params: {
    location: location
    containerRegistryName: container_registry_name
    tags: tags
  }
}

module applicationInsights 'modules/applicationinsights.bicep' = [for deployment_stage in stages:  {
  name: 'appi-${name}-${uniqueSuffix}-deployment-${deployment_stage}'
  params: {
    location: location
    applicationInsightsName: 'appi-${name}-${deployment_stage}-${uniqueSuffix}'
    tags: tags
  }
}]

module azuremlWorkspace 'modules/machinelearning.bicep' = [for (deployment_stage, i) in stages:  {
  name: 'mlw-${name}-${uniqueSuffix}-deployment-${deployment_stage}'
  params: {
    // workspace organization
    machineLearningName: workspace_names[i]
    machineLearningFriendlyName: workspace_names[i]
    machineLearningDescription: 'This is a workspace.'
    location: location
    // prefix: name
    tags: tags

    // dependent resources
    applicationInsightsId: applicationInsights[i].outputs.applicationInsightsId
    containerRegistryId: containerRegistry.outputs.containerRegistryId
    keyVaultId: keyvault[i].outputs.keyvaultId
    storageAccountId: storage[i].outputs.storageId
  }
  dependsOn: [
    keyvault
    containerRegistry
    applicationInsights
    storage
  ]
}]



var info = [for (stage, i) in stages: {
                        '${stage}':{
                          'keyvault': keyvault_names[i]
                          'storage': storage_names[i]
                          'containerregistry': container_registry_name
                          'workspace': workspace_names[i]
                          'subscription': subscription().subscriptionId
                          'resourcegroup': resourceGroup().name
                        }
                      }]

var config = [for (stage, i) in stages: {
                        '${stage}':{
                          'subscription_id': subscription().subscriptionId
                          'resource_group': resourceGroup().name
                          'workspace_name': workspace_names[i]
                        }
                      }]


// more than 2 stages -> make 1 object that combines the first 2 from info, else it is just a list with 1 element and that is returned 
var intermed_results = length(stages) > 1 ? union(info[0],info[1]) : info[0]
// if there are 3 stages then we add the 3rd element of info to the union of objects else return the previous result
var results = length(stages) > 2 ? union(intermed_results, info[2]) : intermed_results

// more than 2 stages -> make 1 object that combines the first 2 from info, else it is just a list with 1 element and that is returned 
var intermed_results1 = length(stages) > 1 ? union(config[0],config[1]) : config[0]
// if there are 3 stages then we add the 3rd element of info to the union of objects else return the previous result
var results1 = length(stages) > 2 ? union(intermed_results, config[2]) : intermed_results1



output deployment_results object = results
output config object = results1

