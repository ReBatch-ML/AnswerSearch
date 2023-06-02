#check if endpoint api key file exist and start local deployment
#run this script inside the GSC_OPEN_BOOK_QA directory
#need a json file endpoint_api/enpoint_api.json with API keys and urls inside of the format:
#{
#   'FLAN_URL' : $flan_ulr$,
#   'FLAN_KEY' : $flan_key$,
#    'DOLLY_URL' : $dolly_ulr$,
#    'DOLLY_KEY' : $dolly_key$,
#    'SEMSEARCH_URL' : $ss_ulr$,
#    'SEMSEARCH_KEY' : $ss_key$
#}
if [ -e "endpoint_api/api_keys.json" ]
then
    echo "endpoint api keys found. Starting local deployment"
    streamlit run src/deployment/web_app/app.py
else
    echo "endpoint api keys not found; make sure you have them in endpoint_api/api_keys.json"
fi