"""Scoring script of the flan search backend"""
import json
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os


def init():
    """Initialize the model and tokenizer when deployed in the cloud
    """
    global flan_model
    global tokenizer

    model_dir = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "flan")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    flan_model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_dir, "model"))
    flan_model = flan_model.to('cuda')


def qa(input_text, temperature=0.7):
    """Invoke model with question and context

    Args:
        input_text (String): String containing the question, the prompt and the context

    Returns:
        String : Answer to the question
    """
    input = tokenizer(input_text, return_tensors='pt').input_ids.to('cuda')
    output = flan_model.generate(input, max_length=400, temperature=temperature)
    return str(tokenizer.decode(output[0], skip_special_tokens=True))


def run(raw_query):
    """Function called by the Azure ML service; it receives the raw json query and returns the answer in a new json query
    """
    logging.info(f"Received the raw query{raw_query}")

    js = json.loads(raw_query)
    context = js['context']
    question = js['question']
    mode = js['mode']
    prompt = js['prompt']
    temperature = js['temperature']
    input_text = ""
    if mode == None or mode == "noprompt":
        prompt = "Only answer if you can answer the question based on the context, otherwise answer Not Found"
        input_text = "context:" + context + "\n" + "question: " + question + "\n" + "Only answer if you can answer the question based on the context, otherwise answer Not Found"

    elif mode == "partialprompt":
        input_text = "context:" + context + "\n" + "question: " + question + "\n" + prompt

    elif mode == "fullprompt":
        input_text = prompt

    elif mode == "2-step-partialprompt":
        input_text_temp = "context:" + context + "\n" + "question: " + question + "\n" + "Can you answer the question based on the context? Answer yes or no:"
        temp = qa(input_text_temp, temperature)
        if temp.lower() == str("no") or temp.lower() == str("no."):
            logging.info(f"Could not answer the question with the given context")
            return {'answer': ["Not Found"]}
        input_text = "context:" + context + "\n" + "question: " + question + "\n" + prompt

    else:
        raise ValueError("Invalid mode")

    anwser = qa(input_text, temperature)
    logging.info(f"Answered the question with the given context if possible")

    return {'answer': [anwser]}


def deprecated_run(raw_query):
    """Function called by the Azure ML service; It queries the model twice, first to check if it can answer the question, then to answer the question
    """

    logging.info(f"Received the raw query{raw_query}")

    js = json.loads(raw_query)
    context = js['context']
    question = js['question']

    input_text_1 = "context:" + context + "\n" + "question: " + question + "\n" + "Can you answer the question based on the context? Answer yes or no:"
    output_1_text = qa(input_text_1)

    #check if the answer was no
    if output_1_text.lower() == str("no") or output_1_text.lower() == str("no."):
        logging.info(f"Could not answer the question with the given context")
        return json.dumps(
            {
                'question': [question],
                'context': [context],
                'answer': ['Could not answer the question with the given context.']
            }
        )

    input_text_2 = "context:" + context + "\n" + "question: " + question + "\n" + "Answer the question based on the context:"
    anwser = qa(input_text_2)
    logging.info(f"Answered the question with the given context")
    return json.dumps({'question': [question], 'context': [context], 'answer': [anwser]})
