"""Streamlit app for the semantic search engine demo"""
import os
from pathlib import Path
import ssl
import json
import yaml
import urllib.request
from PIL import Image
# from time import time
import streamlit as st
from streamlit_authenticator import Authenticate
from azure.keyvault.secrets import SecretClient
from highlighted_text import get_annotated_html
from azure.identity import DefaultAzureCredential
import time

with open("config.yaml", 'r') as file:
    deployment_config = yaml.safe_load(stream=file)

client_name = deployment_config["client_name"]


def login():
    """
    Login page
    """

    # Bit of a hack to get the logo in the center of the page
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("")
    with col2:
        if Path.exists(Path("logo.jpg")):
            st.image(Image.open("logo.jpg"), width=200)
        # st.image(Image.open("logo.jpg"), width=200)
        st.write(f"# Welcome to the {client_name} semantic search engine")
        name, authentication_status, username = authenticator.login('Login', 'main')
        st.session_state["name"] = name
    with col3:
        st.write("")


def get_corpus_names():
    """
    Fetches the corpus names from the config file

    :return: a list of corpus names
    """
    with open("config.yaml", 'r') as file:
        deployment_config = yaml.safe_load(stream=file)

    return [corpus["name"] for corpus in deployment_config["corpora"]]


def demo():
    """
    Main page of the demo after login
    """
    IS_ADMIN_USER = st.session_state["admin"]

    with st.sidebar:
        st.header("Search settings:")
        corpus = st.radio(
            "Choose the corpus to search",
            get_corpus_names(),
            help="In which corpus do you want to search?",
        )

        highlighting = st.checkbox(
            label="Highlight relevant sentences?",
            value=True,
            help="This is a costly operation and will slow down result retrieval."
        )

        default_top_k = 64
        if IS_ADMIN_USER:
            top_k = st.number_input("# results of bi-encoder", min_value=1, max_value=1000, value=default_top_k)
        else:
            top_k = default_top_k

    with center:
        with st.form(key="text_form"):
            query = st.text_input(label=f"Search in {corpus}:", placeholder="Search query...", key="query")
            search_btn = st.form_submit_button(label="Search")

    if not search_btn:
        st.stop()

    def process_result(json_data, key_prefix=""):
        """
        Process the result of the semantic search

        :param json_data: the json data returned by the backend
        :param key_prefix: the prefix of the keys in the json data

        :return: a list of tuples (paragraph, score)
        """
        paragraphs = json_data[key_prefix + 'text']
        scores = json_data[key_prefix + 'cross_encoder_prediction_score']

        results = []
        for idx, paragraph in enumerate(paragraphs):
            results.append((paragraph, scores[idx]))

        return results

    def get_confidence(score):
        """
        Get the confidence level of the result

        :param score: the score of the result
        """
        if score > 0.9:
            return "Very high"
        elif score > 0.8:
            return "High"
        elif score > 0.6:
            return "Medium"
        elif score > 0.3:
            return "Low"
        else:
            return "Very low"

    def write_results(results, extra=False):
        """
        Write the results to the page

        :param results: the results to write
        """
        if extra:
            st.markdown(
                "## Sadly not that many results were found for your year selection... Extra results from other years are added below."
            )
            right.info(
                'Maybe try to be more specific in your query if the document was not in the first results. There might be too much overlap with documents from other years',
                icon="ℹ️"
            )
            # right.write("Maybe try to be more specific in your query if the document was not in the first results. There might be too much overlap with documents from other years")

        for i, (paragraph, score) in enumerate(results, 1):
            with center.container():
                st.markdown(f"#### {i}: Confidence: {get_confidence(score)}")
                if type(paragraph) == list:
                    st.markdown(
                        get_annotated_html(*paragraph),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        get_annotated_html(paragraph),
                        unsafe_allow_html=True,
                    )

    def call_model():
        """
        Call the model to get the results
        """

        def allowSelfSignedHttps(allowed):
            """
            bypass the server certificate verification on client side
            Args:
                allowed (bool): Is it allowed to use self-signed certificate
            """
            if allowed and not os.environ.get('PYTHONHTTPSVERIFY',
                                              '') and getattr(ssl, '_create_unverified_context', None):
                ssl._create_default_https_context = ssl._create_unverified_context

        allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.

        data = {"query": [query], "corpus": corpus, "highlighting": highlighting, "top_k": top_k}

        body = str.encode(json.dumps(data))

        url = API_URL
        # The azureml-model-deployment header will force the request to go to a specific deployment.
        # Remove this header to have the request observe the endpoint traffic rules
        # API_KEY = get_endpoint_api_key()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': ('Bearer ' + API_KEY),
            'azureml-model-deployment': f'semantic-search'
        }

        req = urllib.request.Request(url, body, headers)

        try:
            print("Send request...")
            response = urllib.request.urlopen(req)
            result = response.read().decode('utf8')
            result = json.loads(result)

            # left.write(f"Request time: {end_request-start_request}s")

            return result

        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))

    with st.spinner('Searching for your query...'):
        start = time.time()
        result = call_model()
        end = time.time()

    # result = json.load(open('test.json'))
    processed_results = process_result(result)
    write_results(processed_results)

    prefix = "extra_"
    # check if there are extra documents in the dictionary
    if all(prefix + k in result for k in ["doc_id", "text", "title", "cross_encoder_prediction_score"]):
        extra_processed = process_result(result, key_prefix=prefix)
        write_results(extra_processed, extra=True)

    if IS_ADMIN_USER:
        left.write(f"bi_encoder_time: {result['bi_encoder_time']}")
        left.write(f"faiss_time: {result['faiss_time']}")
        left.write(f"cross_encoder_time: {result['cross_encoder_time']}")
        left.write(f"Sentence_highlighting_time: {result['sentence_time']}")
        left.write(f"filter_time: {result['filter_time']}")
        left.write(f"Total time: {end-start}s")


def get_endpoint_api_info():
    """
    Get the API key for the endpoint
    """
    KVUri = os.getenv('AZURE_KEYVAULT_RESOURCEENDPOINT')
    cred = DefaultAzureCredential()
    vault_client = SecretClient(vault_url=KVUri, credential=cred)

    API_KEY = vault_client.get_secret('api-key').value
    API_URL = vault_client.get_secret('api-url').value

    return API_URL, API_KEY


def _max_width_():
    """
    function that sets the max width of the page
    """
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    API_URL, API_KEY = get_endpoint_api_info()

    logo = Image.open("rebatch_logo.png")

    st.set_page_config(
        page_title=f"Semantic Search {client_name}",
        page_icon=logo,
        layout="wide",
        menu_items=None,
        initial_sidebar_state="auto"
    )

    _max_width_()

    st.session_state["admin"] = False

    with open('auth.yaml') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    left, center, right = st.columns([1, 2, 1])
    with center:
        if Path('logo.jpg').is_file():
            st.image(Image.open("logo.jpg"), width=200)

        st.write(f"# {client_name} semantic search engine")

    authenticator = Authenticate(
        config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'],
        config['preauthorized']
    )

    if "landing" not in st.session_state:
        st.session_state["landing"] = "login"
    landing = st.session_state['landing']

    authentication_status = None
    if st.session_state["landing"] == "login":
        # col1, col2, col3 = st.columns([1, 1, 1])
        with center:
            name, authentication_status, username = authenticator.login('Login', 'main')
            st.session_state["name"] = name
    elif st.session_state["landing"] == "register":
        # col1, col2, col3 = st.columns([1, 1, 1])
        with center:
            try:
                if authenticator.register_user('Register user', preauthorization=True):
                    st.success('User registered successfully')
                    # update the auth.yaml with new user
                    with open('auth.yaml', 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)

            except Exception as e:
                st.error(e)

    swap_page = {"register": "login", "login": "register"}

    if authentication_status:
        # st.session_state["landing"] = "main"
        st.sidebar.write(f'Welcome *{name}*')
        authenticator.logout('Logout', 'sidebar')
        # draw the entire demo
        demo()
    else:
        if authentication_status == False:
            center.error('Username/password is incorrect')
        elif authentication_status is None and st.session_state["landing"] != "register":
            center.warning('Please enter your username and password')

        other_page_name = swap_page[st.session_state["landing"]]

        # button name points toward other page
        with center:
            if st.button(other_page_name + " instead"):
                # set new state to the new page
                st.session_state['landing'] = other_page_name
                st.experimental_rerun()
