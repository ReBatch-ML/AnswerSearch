"""Code where actual embeddings are created and saved as a huggingface dataset"""
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json
import glob
import torch
import os
from datasets import Dataset as HfDataset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from azureml.core import Run, Dataset, Datastore
import argparse
from azureml.core import Run, Dataset, Datastore, Model
from transformers.tokenization_utils_base import BatchEncoding
from sentence_transformers import SentenceTransformer, models


# huggingface maping function that takes a sample and returns a dict
def map_to_hf_dataset(sample):
    """
    Parse all data from the data column into a dict

    Args:
        sample (Huggingface dataset row): The row on which the function is applied

    Returns:
        Huggingface dataset row: The row with the data column parsed into a dict
    """
    string_data = sample["data"].replace("'", '"')

    sample["data"] = json.loads(string_data)
    return sample


class EmbeddingGenerator:
    """
    Class that generates embeddings for articles and saves them as a huggingface dataset
    """

    def __init__(self, model_path):
        """

        :param model_path: path to the model to use
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Device: ", self.device)
        print(f"Loading model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = SentenceTransformer(model_path)

        self.model = self.model.to(self.device)
        self.model_path = model_path
        self.MODEL_NAME = model_path

        self.CLS_TOKEN_ID = self.tokenizer.cls_token_id
        self.SEP_TOKEN_ID = self.tokenizer.sep_token_id
        self.PAD_TOKEN_ID = self.tokenizer.pad_token_id

    def fix_tokens_and_mask(self, tokens, mask):
        """
        Fix sentences to be of the same length, inluding the CLS and SEP tokens
        Also update the mask to have the same length 
        IMPORTANT: THIS FUNCTION ALWAYS ADDS THE CLS AND SEP TOKENS SO THE MAX POTENTIAL LENGTH OF THE SENTENCE WILL BE MAX_LENGTH + 2
        SO ENSURE THAT THE MAX_LENGTH IS SMALLER THAN THE MAX LENGTH OF THE MODEL - 2

        :param tokens: the tokens to fix
        :param mask: the mask to fix

        :return: the fixed tokens and mask in a BatchEncoding object, so that it can be put on the GPU later, which isn't possible for a regular Dict
        """
        tokens = torch.cat([torch.tensor([self.CLS_TOKEN_ID]), tokens, torch.tensor([self.SEP_TOKEN_ID])])
        # also important in case sentence matches max length after adding the cls and sep tokens
        # this was not done before and that would result in tokens and mask not having the same length
        mask = torch.cat([torch.tensor([0]), mask, torch.tensor([0])])

        if len(tokens) < MAX_LENGTH:
            tokens = torch.cat([tokens, torch.tensor([self.PAD_TOKEN_ID] * (MAX_LENGTH - len(tokens)))])
            mask = torch.cat([mask, torch.tensor([0] * (MAX_LENGTH - len(mask)))])
        # return BatchEncoding({"input_ids": tokens, "attention_mask": mask})
        return {"input_ids": tokens, "attention_mask": mask}

    def generate_embeddings(self, corpus):
        """
        Generates embeddings for articles and also saves them as a huggingface dataset.

        :param articles: the articles to generate embeddings for
        """

        dataset_list = []
        print("Split embeddings into chunks of 512 tokens")
        with open(corpus, 'r') as jsonl_file:
            for line in tqdm(jsonl_file, desc="Splitting articles into chunks"):
                entry = json.loads(line)
                text = entry["text"]

                tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)

                # if orginal text is too short, skip
                if len(text.split(' ')) < 4:
                    continue

                for (input_tokens, mask) in zip(
                    tokens["input_ids"].squeeze().split(SPLIT_LENGTH),
                    tokens["attention_mask"].squeeze().split(SPLIT_LENGTH)
                ):

                    t = self.tokenizer.decode(input_tokens.squeeze())

                    # Skip if the splitted text is too short too
                    if len(t.split(' ')) < 4:
                        continue

                    d = {
                        "id": entry['id'],
                        "cord_uid": entry['cord_uid'],
                        "text": t,
                        "data": self.fix_tokens_and_mask(input_tokens, mask),
                        "title": entry['title'],
                        "url": entry['url'],
                    }

                    dataset_list.append(d)
                    # incrementally write to csv to save memory
                    # if not Path('data.csv').exists():
                    #     writeheader = True
                    # else:
                    #     writeheader = False
                    # with open('data.csv', 'a') as f:
                    #     w = DictWriter(f, fieldnames=d.keys())
                    #     if writeheader:
                    #         w.writeheader()
                    #     w.writerow(d)

                    assert len(d["data"]["attention_mask"]) <= MAX_LENGTH, "Input tokens too long"

        print("Create huggingface dataset")
        # dataset = load_dataset("csv", data_files="data.csv")
        # dataset = dataset.map(map_to_hf_dataset, num_proc=4)

        dataset = HfDataset.from_list(dataset_list)

        print("Map tokens to embeddings")
        with torch.no_grad():

            dataset = dataset.map(
                lambda x: {
                    "embedding":
                        self.model(
                            BatchEncoding(
                                {key: torch.tensor([d[key] for d in x["data"]]) for key in x["data"][0].keys()}
                            ).to(self.device)
                        )["sentence_embedding"].cpu().numpy()
                },
                remove_columns=["data"],
                batched=True,
                batch_size=2000
            )

        # save embeddings to disk
        save_folder = f"processed/{os.path.basename(file).split('.')[0]}"
        dataset.save_to_disk(save_folder)

        # save model info
        json.dump({
            "model_name": self.MODEL_NAME,
        }, open(f"{save_folder}/model_info.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="The dataset to use")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size to use")
    parser.add_argument("--output_folder", type=str, help="Place where to save the embeddings to")
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of the input")
    parser.add_argument("--split_length", type=int, default=512, help="The length to split the input into")
    parser.add_argument("--extra_margin", type=int, default=20, help="The extra margin to add to the split length")
    parser.add_argument("--save_on_azure", action="store_true", help="Save the embeddings on azure")
    parser.add_argument("--model_name", type=str, help="Name of model", default="")
    parser.add_argument("--model_version", type=int, help="Version of the model in the registry", default=-1)
    args = parser.parse_args()

    # Load model from HuggingFace Hub
    MAX_LENGTH = args.max_length
    SPLIT_LENGTH = args.split_length
    EXTRA_MARGIN = args.extra_margin

    run = Run.get_context()
    workspace = run.experiment.workspace

    # if the model version is a positive number then it should be a model from the registry
    # so download it first, else it should be a model name from the huggingface hub
    if args.model_version > -1:
        model_path = Model.get_model_path(model_name=args.model_name, version=args.model_version, _workspace=workspace)
        embedding_generator = EmbeddingGenerator(model_path)
    else:

        embedding_generator = EmbeddingGenerator(args.model_name)

    for file in glob.glob(f'{args.dataset}/*.jsonl'):
        # articles = json.load(open(file, "r"))
        print(f"Processing {file}")
        embedding_generator.generate_embeddings(file)
        print("=========================")

    if args.save_on_azure:
        # Upload to storage
        # Gets the default datastore, this is where we are going to save our new data
        datastore = Datastore.get(workspace, 'workspaceblobstore')
        out_dir_path = "processed"

        # Select the directory where we put our processed data and upload it to our datastore
        azure_save_dir = args.output_folder

        preprocessed = Dataset.File.upload_directory(out_dir_path, (datastore, azure_save_dir), overwrite=True)

        # Clean up and remove all local data after we have uploaded the data to the datastore
        shutil.rmtree(out_dir_path)
