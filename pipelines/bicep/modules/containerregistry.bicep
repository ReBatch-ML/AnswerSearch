// Creates an Azure Container Registry with Azure Private Link endpoint
@description('Azure region of the deployment')
param location string

@description('Tags to add to the resources')
param tags object

@description('Container registry name')
param containerRegistryName string

// @description('Container registry private link endpoint name')
// param containerRegistryPleName string

// @description('Resource ID of the subnet')
// param subnetId string

// @description('Resource ID of the virtual network')
// param virtualNetworkId string

var containerRegistryNameCleaned = replace(containerRegistryName, '-', '')

// var privateDnsZoneName = 'privatelink${environment().suffixes.acrLoginServer}'

// var groupName = 'registry' 

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2021-09-01' = {
  name: containerRegistryNameCleaned
  location: location
  tags: tags
  sku: {
    name: 'Premium'
  }
  properties: {
    adminUserEnabled: true
    dataEndpointEnabled: false
    networkRuleBypassOptions: 'AzureServices'
    // networkRuleSet: {
    //   defaultAction: 'Deny'
    // }
    policies: {
      quarantinePolicy: {
        status: 'disabled'
      }
      retentionPolicy: {
        status: 'enabled'
        days: 7
      }
      trustPolicy: {
        status: 'disabled'
        type: 'Notary'
      }
    }
    publicNetworkAccess: 'Enabled'
    zoneRedundancy: 'Disabled'
  }
}

output containerRegistryId string = containerRegistry.id
output name string = containerRegistry.name
