""" UI of semantic search and Question Answering with Flan"""
from email.policy import default
import os
import ssl
import json
import yaml
import numpy as np
import urllib.request
from PIL import Image
from time import time, sleep
import streamlit as st
from streamlit_authenticator import Authenticate
from azure.keyvault.secrets import SecretClient
from highlighted_text import get_annotated_html
from azure.identity import DefaultAzureCredential


def get_api_keys():
    """Return API urls and keys for the different services; also returns the path to the web_app folder (which is needed for local deployment)
        When deployed locally, the keys are read from a local file. 
        When deployed in Azure, the keys are read from the Key Vault
    Returns:
        String: URLs and keys for SEMSEARCH, FLAN and DOLLY
    """
    path = ''
    KVUri = os.getenv('AZURE_KEYVAULT_RESOURCEENDPOINT')
    if KVUri == None:
        #local deployment:
        path = 'src/deployment/web_app/'
        print("No Key Vault found, using local keys")
        key_file = json.load(open('endpoint_api/api_keys.json'))
        SEMSEARCH_URL = key_file['SEMSEARCH_URL']
        SEMSEARCH_KEY = key_file['SEMSEARCH_KEY']
        FLAN_URL = key_file['FLAN_URL']
        FLAN_KEY = key_file['FLAN_KEY']
        DOLLY_URL = key_file['DOLLY_URL']
        DOLLY_KEY = key_file['DOLLY_KEY']
    else:
        #azure deployment:
        cred = DefaultAzureCredential()
        vault_client = SecretClient(vault_url=KVUri, credential=cred)
        SEMSEARCH_KEY = vault_client.get_secret('semsearch-key').value
        SEMSEARCH_URL = vault_client.get_secret('semsearch-url').value
        FLAN_KEY = vault_client.get_secret('qa-key').value
        FLAN_URL = vault_client.get_secret('qa-url').value
        DOLLY_KEY = vault_client.get_secret('dolly-key').value
        DOLLY_URL = vault_client.get_secret('dolly-url').value
    return SEMSEARCH_URL, SEMSEARCH_KEY, FLAN_URL, FLAN_KEY, DOLLY_URL, DOLLY_KEY, path


SEMSEARCH_URL, SEMSEARCH_KEY, FLAN_URL, FLAN_KEY, DOLLY_URL, DOLLY_KEY, path = get_api_keys()
ANSWERS_AMOUNT = 5
QUERY_AMOUNT = 64
online_deployment = "semantic-search"
qa_deployment = "flan"
dolly_deployment = "dolly"

logo = Image.open(path + "cronos_logo.webp")

st.set_page_config(page_title="Rebatch Knowledge Assistant", page_icon=logo, layout="wide")


def _max_width_():
    """
        Function that sets the max width
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


_max_width_()

with open(path + 'auth.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = Authenticate(
    config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'],
    config['preauthorized']
)
left, center, right = st.columns([1, 2, 1])

if "landing" not in st.session_state:
    landing = "login"
else:
    landing = st.session_state['landing']

if landing == "login":
    name, authentication_status, username = authenticator.login('Login', 'main')
elif landing == "register":
    authentication_status = None
    try:
        if authenticator.register_user('Register user', preauthorization=True):
            st.success('User registered successfully')
            # update the auth.yaml with new user
            with open(path + 'auth.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)

swap_page = {"register": "login", "login": "register"}


def demo():
    """
    Function that draws the entire demo
    """
    IS_ADMIN_USER = authenticator.credentials["usernames"][username]["email"].endswith("@rebatch.be")
    query = ""
    prompt = ""
    prompt_option = "noprompt"
    dolly_prompt = "Summarize the search results into one coherent answer, based on the question:"

    with st.sidebar:
        st.header("Search settings:")
        corpus = st.radio(
            "Choose the corpus to search",
            ["Public Registry"],
            help="In which corpus of documents do you want to search?",
        )
        if corpus == "Public Registry":
            years = st.select_slider(label="Years to include:", options=range(1993, 2023), value=(1993, 2022))
        

        default_top_k = 128
        if IS_ADMIN_USER:
            highlighting = st.checkbox(
                label="Highlight relevant sentences?",
                value=True,
                help="This is a costly operation and will slow down result retrieval."
            )
            top_k = st.number_input("# results of bi-encoder", min_value=1, max_value=1000, value=default_top_k)
            prompt_option = st.radio(
                label="Select prompt customization options",
                options=["noprompt", "fullprompt", "partialprompt", "2-step-partialprompt"],
                index=3,
                help=
                "Select how to query the flan model. 'noprompt': the prompt engineering is done in the backend; 'fullprompt': the chosen prompt goes directly to the flan backend and the semantic search is not queried for a context; 'partialprompt': the context is still searched for but you can chose a prompt to add to the question; '2-step-partialprompt': same as partialprompt but first the model is asked if it can answer the question with given context."
            )
            temperature = st.slider(label="Temperature", min_value=0.001, max_value=1.0, value=0.01, step=0.01)

        else:
            highlighting = True
            top_k = default_top_k
            prompt = "Generate a comprehensive and informative answer based on the context that answers the question. Think step-by-step:"
            prompt_option = "2-step-partialprompt"
            temperature = 0.01

    with center:
        st.title("Rebatch Knowledge Assistant")
        # st.header("")
        if IS_ADMIN_USER:
            with st.form(key="text_form"):
                if prompt_option == "fullprompt":
                    prompt = st.text_area(
                        label=f"Search in given prompt:", placeholder="Process prompt...", key="query"
                    )

                elif prompt_option == "partialprompt" or prompt_option == "2-step-partialprompt":
                    query = st.text_input(label=f"Search in {corpus}:", placeholder="Search query...", key="query")
                    prompt = st.text_area(
                        label=f"Add prompt to question:",
                        value=
                        "Generate a comprehensive and informative answer based on the context that answers the question. Think step-by-step:",
                        key="prompt"
                    )
                    dolly_prompt = st.text_area(
                        label=f"Add dolly prompt to summarize the answers:", value=dolly_prompt, key="dollyprompt"
                    )

                else:
                    query = st.text_input(label=f"Search in {corpus}:", placeholder="Search query...", key="query")
                    dolly_prompt = st.text_area(
                        label=f"Add dolly prompt to summarize the answers:", value=dolly_prompt, key="dollyprompt"
                    )
                search_btn = st.form_submit_button(label="Search")
        else:
            with st.form(key="text_form"):
                query = st.text_input(label=f"Search in {corpus}:", placeholder="Search query...", key="query")
                search_btn = st.form_submit_button(label="Search")

    if not search_btn:
        st.stop()

    def process_result(json_data, key_prefix=""):
        """
        Function that will process results retrieved from the backend, order them by doc_id etc

        Args:
            json_data (Dict): Data returned from backend
            key_prefix (str, optional): Needed in case there's extra results to pad the results of a strict filter. Defaults to "".

        Returns:
            List(Tuple): List of paragraphs (and metadata) belonging to each document
        """
        print(json_data.keys())
        doc_ids = np.array(json_data[key_prefix + 'doc_id'])
        paragraphs = np.array(json_data[key_prefix + 'text'])
        scores = np.array(json_data[key_prefix + 'cross_encoder_prediction_score'])
        contexts = np.array(json_data[key_prefix + 'context'])
        # scores = np.around(scores, 2)

        paragraphs_per_doc = []
        for t in np.unique(doc_ids):
            idxs = np.argwhere(doc_ids == t)
            # right.write(json_data.keys())
            rank = min(idxs)[0]
            try:
                assert idxs.tolist() == sorted(idxs)
            except AssertionError as e:
                print(f"Paragraphs for document {t} aren't sorted properly for each document: {e}")
                right.write(f"Paragraphs for document {t} aren't sorted properly for each document: {e}")
            # right.write(f"rank: {rank}")
            doc_id = json_data[key_prefix + "doc_id"][rank].replace(' ', '-')
            paragraphs_per_doc.append(
                (rank, json_data[key_prefix + 'title'][rank], doc_id, paragraphs[idxs], scores[idxs], contexts[idxs])
            )
            # paragraphs_per_title[t] = paragraphs[idxs]

        # print(paragraphs_per_title)
        paragraphs_per_doc = sorted(paragraphs_per_doc, key=lambda tup: tup[0])
        return paragraphs_per_doc

    def url_reachable(url):
        """
        Check if an url is reachable. Used to verify if a pdf URL points to a valid place.

        Args:
            url (str): url to check

        Returns:
            Bool: True of status is 200 else False
        """
        status = urllib.request.urlopen(url=url).getcode()

        return status == 200

    def get_confidence(score):
        """
        Function that converts scores into confidence intervals.

        Args:
            score (float): Score

        Returns:
            str: Name of confidence interval
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

    def write_answers(results):
        """
        Calls the flan model and writes the answers in dropdown lists with their context.

        Args:
            results (List[Tuple])): Processed ordered from most to least confident documents
        """
        answers = []
        answer_count = 0
        query_count = 0
        for _, title, doc_id, paragraphs, scores, contexts in results:
            label = f"{title}"

            for i, c in enumerate(contexts):
                if answer_count >= ANSWERS_AMOUNT or query_count >= QUERY_AMOUNT:
                    break
                if c == "DUPLICATE":
                    continue
                #add title to context to give more information
                query_count += 1
                context = title + "\n" + c[0]
                if get_confidence(scores[i][0]) != 'Very low':
                    answer = call_qa_model(context)
                    if answer == None:
                        continue  #if request timeout, continue with next context
                    answer = np.array(answer['answer'])[0]
                    if answer != 'Not Found':
                        answers.append(answer)
                        answer_count += 1
                        with st.expander(label=f"Answer: {answer}"):
                            st.markdown(f" #### Context found in:  {label}")
                            if corpus == "Public Registry":
                                url = f"https://data.consilium.europa.eu/doc/document/{doc_id}/en/pdf"
                                if url_reachable:
                                    link = f'[pdf]({url})'
                                else:
                                    st.write("no pdf available")
                                st.markdown(link, unsafe_allow_html=True)
                            st.markdown("#### Context:")
                            st.markdown(context)
                            st.markdown(f"#### Answer: ")
                            st.markdown(answer)
        return answers

    def write_complete_answer(answers):
        """Write the final answer using the dolly model.

        Args:
            answers (List): List of answers to summarize
        """
        result = call_dolly_model(answers)
        if result == None:
            st.markdown("Sorry, we couldn't find a summarized answer to your question.")
            return
        complete_answer = np.array(result['result'])[0]
        st.markdown(complete_answer)

    def write_results(results, extra=False):
        """
        Writes the processed paragraphs in dropdown lists with potential highlighting.

        Args:
            results (List[Tuple])): Processed results ordered from most to least confident documents
            extra (bool, optional): Whether it's actual results or padded results because of a too strict filter. Defaults to False.
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

        for _, title, doc_id, paragraphs, scores, contexts in results:
            max_score = np.max(scores)
            num_hits = len(paragraphs)
            label = f"{title} ---- SEARCH STATS: {num_hits} hits - confidence: {get_confidence(max_score)}"
            with st.expander(label=label):
                # st.write(paragraphs)

                if corpus == "Public Registry":
                    url = f"https://data.consilium.europa.eu/doc/document/{doc_id}/en/pdf"
                    if url_reachable:
                        link = f'[pdf]({url})'
                    else:
                        st.write("no pdf available")
                    st.markdown(link, unsafe_allow_html=True)

                for i, p in enumerate(paragraphs):

                    st.markdown(f"### Paragraph {i + 1} with confidence: {get_confidence(scores[i][0])}")

                    par = p[0]
                    if type(par) == list:
                        st.markdown(
                            get_annotated_html(*par),
                            unsafe_allow_html=True,
                        )

                    else:
                        st.markdown(
                            get_annotated_html(par),
                            unsafe_allow_html=True,
                        )

    def call_model(url, data, headers):
        """
        Function that calls the backend
        
        Returns:
           Dict: Result of the backend 
        """

        def allowSelfSignedHttps(allowed):
            # bypass the server certificate verification on client side
            if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
                ssl._create_default_https_context = ssl._create_unverified_context

        allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.


        body = str.encode(json.dumps(data))

        start_request = time()
        req = urllib.request.Request(url, body, headers)

        try:
            print("Send request...")
            response = urllib.request.urlopen(req)
            result = response.read().decode('utf8')
            result = json.loads(result)
            end_request = time()

            # left.write(f"Request time: {end_request-start_request}s")

            return result

        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))

    def call_qa_model(text=""):
        """
        Function that calls the flan backend
        
        Returns:
           Dict: Result of the flan backend 
        """

        data = {'question': query, 'prompt': prompt, 'context': text, 'mode': prompt_option, 'temperature': temperature}
        headers = {
            'Content-Type': 'application/json',
            'Authorization': ('Bearer ' + FLAN_KEY),
            'azureml-model-deployment': f'{qa_deployment}'
        }
        return call_model(FLAN_URL, data, headers)

    def call_dolly_model(answers):
        """
        Function that calls the dolly backend
        
        Returns:
           Dict: Result of the dolly backend 
        """

        data = {'question': query, 'answers': answers, 'prompt': dolly_prompt, 'temperature': temperature}

        headers = {
            'Content-Type': 'application/json',
            'Authorization': ('Bearer ' + DOLLY_KEY),
            'azureml-model-deployment': f'{dolly_deployment}'
        }
        return call_model(DOLLY_URL, data, headers)

    def call_ss_model():
        """
        Function that calls the semsearch backend
        
        Returns:
           Dict: Result of the semantic search backend 
        """

        data = {
            "query": [query],
            "corpus": corpus,
            "years": list(range(years[0], years[1] + 1)) if corpus == "Public Registry" else None,
            "highlighting": highlighting,
            "top_k": top_k
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': ('Bearer ' + SEMSEARCH_KEY),
            'azureml-model-deployment': f'{online_deployment}'
        }

        return call_model(SEMSEARCH_URL, data, headers)

    if prompt_option == "fullprompt":
        with st.spinner('Searching for your query...'):
            start = time()
            answer = call_qa_model()
            end = time()
            answer = np.array(answer['answer'])[0]
            st.markdown("####  Answer")
            st.markdown(answer)

    else:
        if len(query) == 0:
            st.error("Please enter a question.")
            return

        #initialize containers:
        complete_answer_container = st.empty()
        complete_answer_container.empty()

        answer_container = st.empty()
        answer_container.empty()

        ss_container = st.empty()
        ss_container.empty()

        with st.spinner('Searching for your query...'):
            start = time()
            result = call_ss_model()
            end = time()

        if query[-1] != '?':
            query = query + '?'  # add question mark if not present)]

        
        with complete_answer_container:
            st.markdown("## Final Answer")
        with answer_container:
            st.markdown("## Generated Answers")
        with ss_container:
            st.markdown("## Semantic Search Results")

        processed_results = process_result(result)
        #write SGC results:
        with ss_container.container():
            st.markdown("## Semantic Search Results")
            if result==None:
                st.error("No results")
            write_results(processed_results)
            prefix = "extra_"
            # check if there are extra documents in the dictionary
            if all(prefix + k in result for k in ["doc_id", "text", "title", "cross_encoder_prediction_score"]):
                extra_processed = process_result(result, key_prefix=prefix)
                write_results(extra_processed, extra=True)
        #write answers:
        with answer_container.container():
            st.markdown("## Intermediate Answers")
            with st.spinner('Generating intermediate answers...'):
                answers = write_answers(processed_results)
        #write complete answer:
        with complete_answer_container.container():
            st.markdown("## Final Answer")
            with st.spinner('Summarizing answers...'):
                if len(answers) > 0:
                    write_complete_answer(answers)
                else:
                    st.markdown("No Answers found. Please try another query.")

        if IS_ADMIN_USER:
            left.write(f"bi_encoder_time: {result['bi_encoder_time']}")
            left.write(f"search_time: {result['search_time']}")
            left.write(f"cross_encoder_time: {result['cross_encoder_time']}")
            left.write(f"Sentence_highlighting_time: {result['sentence_time']}")
            left.write(f"filter_time: {result['filter_time']}")
            left.write(f"context_time: {result['context_time']}")
            left.write(f"Total time: {end-start}s")


if authentication_status:
    st.sidebar.write(f'Welcome *{name}*')
    authenticator.logout('Logout', 'sidebar')
    # draw the entire demo
    demo()
else:
    if authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

    other_page_name = swap_page[landing]
    # button name points toward other page
    if st.button(other_page_name + " instead"):
        # set new state to the new page
        st.session_state['landing'] = other_page_name
        st.experimental_rerun()
