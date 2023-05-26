client_name=$(yq -r .client_name configs/deployment_config.yaml)
stages=$(yq -r .deployment_stages configs/deployment_config.yaml)

resource_group_name=${client_name}SemanticSearch

# create resourcegroup
echo "Create resource group ${resource_group_name}"
az group create -l westeurope -n $resource_group_name

# echo "Create service principal"
# service_principal=$(az ad sp create-for-rbac --name SP_${resource_group_name} \
#                          --role contributor \
#                          --scopes /subscriptions/9705804d-0f2b-43de-a254-038eaf5c9255/resourceGroups/${resource_group_name})

# echo "Write Service principal to file"
# echo $service_principal > .cloud/.azure/AZURE_SERVICE_PRINCIPAL.json

echo "Build bicep resources"
deployment_res=$(az deployment group create --resource-group $resource_group_name --template-file pipelines/bicep/resources.bicep --parameters clientName=$client_name deployment_stages="$stages") # service_principal="$service_principal")

# write info about created resources to a file
created_resources=$(jq .properties.outputs.deployment_results.value <<< $deployment_res)
echo $created_resources > .cloud/.azure/resources_info.json

# TODO: create dev/uat/prod_config.json files from deployment_res
envs=$(jq -r '. | keys[]' <<< $created_resources)

i=0
for env in $envs  
do
    jq -r .properties.outputs.config.value.${env} <<< "$deployment_res" > .cloud/.azure/${env}_config.json
    
done

# add the service principal info to each keyvault
# for env in $envs 
# do
#     kv_name=$(jq -r .properties.outputs.deployment_results.value.${env}.keyvault <<< "$deployment_res")
#     echo $kv_name
#     az keyvault secret set --vault-name "$kv_name" --name "AZURE-SERVICE-PRINCIPAL" --value "$service_principal"
# done


