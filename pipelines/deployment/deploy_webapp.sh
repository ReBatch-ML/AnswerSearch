az login --tenant rebatch.onmicrosoft.com

cur_dir=$PWD
app_name="AnswerSearch"
keyvault_name="answerdev8556491192"

# change workdir to webapp folder
cd src/deployment/web_app

# write api key for the endpoint down in file so that it get deployed together with the app
# az ml online-endpoint get-credentials -n online-endpoint -o tsv --query primaryKey -g Semantic_Search -w SemanticSearch_TRAIN > api_key.txt
# deploy the app
az webapp up --runtime PYTHON:3.8 --os-type Linux --resource-group AnswerSearch --sku B1 --location "West Europe" --name $app_name


# identity=$(az webapp identity assign --name $app_name --resource-group Semantic_Search)
# echo $identity
# sp_id=$( jq '.principalId' <<< "${identity}")
# az keyvault set-policy --name semanticsearch7128345119 --object-id $sp_id --secret-permissions get list

# set start.sh as startup script
az webapp config set --resource-group AnswerSearch --name $app_name --startup-file start.sh

az webapp connection create keyvault -g AnswerSearch -n $app_name --tg AnswerSearch --vault $keyvault_name --system-identity --client-type python --connection keyvault_connection

# rm api_key.txt
cd $cur_dir
