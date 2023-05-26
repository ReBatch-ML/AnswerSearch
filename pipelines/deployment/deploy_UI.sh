# appId=$(jq -r .appId .cloud/.azure/AZURE_SERVICE_PRINCIPAL.json)
# password=$(jq -r .password .cloud/.azure/AZURE_SERVICE_PRINCIPAL.json)
# tenant=$(jq -r .tenant .cloud/.azure/AZURE_SERVICE_PRINCIPAL.json)

# az login --service-principal -u $appId -p $password --tenant $tenant

client_name=$(yq -r .client_name "configs/deployment_config.yaml")
cur_dir=$PWD
app_name="semantic-search-${client_name}"
resource_group=${client_name}SemanticSearch
keyvault_name=$(jq -r .dev.keyvault .cloud/.azure/resources_info.json)


cp configs/deployment_config.yaml src/deployment/web_app/config.yaml

# change workdir to webapp folder
cd src/deployment/web_app

# deploy the app
az webapp up --runtime PYTHON:3.8 --os-type Linux --resource-group $resource_group --sku B1 --location "West Europe" --name $app_name

# set start.sh as startup script
az webapp config set --resource-group $resource_group --name $app_name --startup-file start.sh

az webapp connection create keyvault -g $resource_group -n $app_name --tg $resource_group --vault $keyvault_name --system-identity --client-type python --connection keyvault_connection

# rm api_key.txt
cd $cur_dir

# delete the copied file again
rm src/deployment/web_app/config.yaml

