"""Scoring script of the flan search backend"""
import json
import logging
import re
from typing import List
import os
import numpy as np
from transformers import Pipeline, PreTrainedTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def init():
    """Initialize the model and tokenizer when deployed in the cloud
    """
    global dolly_model
    global tokenizer
    global generate_text
    torch.cuda.is_available()

    model_dir = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "dolly")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"), padding_side="left")
    dolly_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, "model"), device_map="auto", torch_dtype=torch.bfloat16
    )
    generate_text = InstructionTextGenerationPipeline(model=dolly_model, tokenizer=tokenizer, max_new_tokens=512)


def list_of_answers_to_string(answer_list):
    """Converts a list of answers to a string with each answer on a new line
    """
    answer_string = ""
    for answer in answer_list:
        answer_string = answer_string + "\n" + answer
    return answer_string


def run(raw_query):
    """Function called by the Azure ML service; it receives the raw json query and returns the answer in a new json query
    """
    logging.info(f"Received the raw query{raw_query}")

    js = json.loads(raw_query)
    answers = js['answers']
    prompt = js['prompt']
    question = js['question'] + "\n"
    temperature = js['temperature']
    llm_text = """
    Question: {}
    Search Results:{}
    {}
    """.format(question, list_of_answers_to_string(answers), prompt)

    res = generate_text(llm_text, temperature=temperature)
    result = res[0]["generated_text"]
    print("ANSWER----------------")
    print(result)

    return {'result': [result]}


class InstructionTextGenerationPipeline(Pipeline):
    """class for text generation pipeline

    Args:
        Pipeline : pipeline class from transformers
    """

    def __init__(
        self, *args, do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs
    ):
        """Initialize the pipeline

        Args:
            do_sample (bool, optional): Whether or not to use sampling. Defaults to True.
            max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 128.
            top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with
                probabilities that add up to top_p or higher are kept for generation. Defaults to 0.92.
            top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
                Defaults to 0.
        """
        super().__init__(*args, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)

    def _sanitize_parameters(self, return_full_text: bool = None, **generate_kwargs):
        """Sanitize and validate the input parameters"""
        preprocess_params = {}

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (token for token in self.tokenizer.additional_special_tokens if token.startswith(RESPONSE_KEY)), None
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(self.tokenizer, tokenizer_response_key)
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {"response_key_token_id": response_key_token_id, "end_key_token_id": end_key_token_id}

        if return_full_text is not None:
            postprocess_params["return_full_text"] = return_full_text

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, instruction_text, **generate_kwargs):
        """preprocess

        Args:
            instruction_text (str): instruction text

        Returns:
            inputs
        """
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction_text)
        # print(prompt_text)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        inputs["prompt_text"] = prompt_text
        inputs["instruction_text"] = instruction_text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        """forward

        Args:
            model_inputs (): model inputs

        Returns:
            dictionalry with generated_sequence and input_ids
        """
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)

        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )

        out_b = generated_sequence.shape[0]
        generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])

        instruction_text = model_inputs.pop("instruction_text")
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "instruction_text": instruction_text}

    def postprocess(self, model_outputs, response_key_token_id, end_key_token_id, return_full_text: bool = False):
        """postprocess"""
        generated_sequence = model_outputs["generated_sequence"][0]
        instruction_text = model_outputs["instruction_text"]

        generated_sequence: List[List[int]] = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:

            # The response will be set to this variable if we can identify it.
            decoded = None

            # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
            if response_key_token_id and end_key_token_id:
                # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
                # prompt, we should definitely find it.  We will return the tokens found after this token.
                try:
                    response_pos = sequence.index(response_key_token_id)
                except ValueError:
                    logger.warn(f"Could not find response key {response_key_token_id} in: {sequence}")
                    response_pos = None

                if response_pos:
                    # Next find where "### End" is located.  The model has been trained to end its responses with this
                    # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                    # this token, as the response could be truncated.  If we don't find it then just return everything
                    # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                    try:
                        end_pos = sequence.index(end_key_token_id)
                    except ValueError:
                        end_pos = None

                    decoded = self.tokenizer.decode(sequence[response_pos + 1:end_pos]).strip()

            if not decoded:
                # Otherwise we'll decode everything and use a regex to find the response and end.

                fully_decoded = self.tokenizer.decode(sequence)

                # The response appears after "### Response:".  The model has been trained to append "### End" at the
                # end.
                m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL)

                if m:
                    decoded = m.group(1).strip()
                else:
                    # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                    # return everything after "### Response:".
                    m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                    if m:
                        decoded = m.group(1).strip()
                    else:
                        logger.warn(f"Failed to find response in:\n{fully_decoded}")

            # If the full text is requested, then append the decoded text to the original instruction.
            # This technically isn't the full text, as we format the instruction in the prompt the model has been
            # trained on, but to the client it will appear to be the full text.
            if return_full_text:
                decoded = f"{instruction_text}\n{decoded}"

            rec = {"generated_text": decoded}

            records.append(rec)

        return records


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        RuntimeError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]
