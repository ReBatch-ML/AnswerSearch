// Creates a storage account, private endpoints and DNS zones
@description('Azure region of the deployment')
param location string

@description('Tags to add to the resources')
param tags object

@description('Name of the storage account')
param storageName string

// @description('Name of the storage blob private link endpoint')
// param storagePleBlobName string

// // @description('Name of the storage file private link endpoint')
// // param storagePleFileName string

// // @description('Resource ID of the subnet')
// // param subnetId string

// // @description('Resource ID of the virtual network')
// // param virtualNetworkId string

@allowed([
  'Standard_LRS'
  'Standard_ZRS'
  'Standard_GRS'
  'Standard_GZRS'
  'Standard_RAGRS'
  'Standard_RAGZRS'
  'Premium_LRS'
  'Premium_ZRS'
])

@description('Storage SKU')
param storageSkuName string = 'Standard_LRS'

var storageNameCleaned = replace(storageName, '-', '')

// var blobPrivateDnsZoneName = 'privatelink.blob.${environment().suffixes.storage}'

// var filePrivateDnsZoneName = 'privatelink.file.${environment().suffixes.storage}'

resource storage 'Microsoft.Storage/storageAccounts@2021-09-01' = {
  name: storageNameCleaned
  location: location
  tags: tags
  sku: {
    name: storageSkuName
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowCrossTenantReplication: false
    allowSharedKeyAccess: true
    encryption: {
      keySource: 'Microsoft.Storage'
      requireInfrastructureEncryption: false
      services: {
        blob: {
          enabled: true
          keyType: 'Account'
        }
        file: {
          enabled: true
          keyType: 'Account'
        }
        queue: {
          enabled: true
          keyType: 'Service'
        }
        table: {
          enabled: true
          keyType: 'Service'
        }
      }
    }
    isHnsEnabled: false
    isNfsV3Enabled: false
    keyPolicy: {
      keyExpirationPeriodInDays: 7
    }
    largeFileSharesState: 'Disabled'
    minimumTlsVersion: 'TLS1_2'
    // networkAcls: {
    //   bypass: 'AzureServices'
    //   defaultAction: 'Deny'
    // }
    supportsHttpsTrafficOnly: true
  }
}

output storageId string = storage.id
output name string = storage.name
