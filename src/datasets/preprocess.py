"""Script that performs the preprocessing of the copora."""
import pandas as pd
# from glob import glob
import json
from tqdm import tqdm
import re
from pathlib import Path
from azureml.core import Run, Dataset, Datastore
import shutil
import argparse
import os


class Preprocessor:
    """
    Class that handles the preprocessing of the corpus.
    """

    def __init__(self, output_location):
        p = Path(output_location)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.output_location = output_location

    def create_corpus(self, metadata):
        """
        Modify this function to work with the data for the project. Metadata file could be replaced with something else if necessary.
        Creates the corpus from the given data. Writes a jsonl file with the following structure:
        {
            "id": "cord_uid_index",
            "cord_uid": "cord_uid",
            "paper_id": "paper_id",
            "title": "title",
            "url": "url",
            "text": "paragraph text"
        }
        Args:
            metadata (pd.DataFrame): Dataframe that contains the metadata and locations of all the files.
        """
        counter = 0
        with open(self.output_location, 'w') as f:
            for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
                pmc_location = row.pmc_json_files

                #  ensure pmc is not nan
                if isinstance(pmc_location, str):
                    file_location = pmc_location
                else:
                    continue

                # does file exist?
                if Path(file_location).is_file():
                    content = json.load(open(file_location))
                else:
                    continue

                entry = {}
                for idx, paragraph in enumerate(content['body_text']):
                    entry['id'] = f"{row['cord_uid']}_{idx}"
                    entry['cord_uid'] = row['cord_uid']
                    entry["paper_id"] = content['paper_id']
                    entry['title'] = content['metadata']['title']
                    entry['url'] = row['url']

                    text = paragraph['text']

                    # always lists so can be appended
                    citations = paragraph['cite_spans']
                    references = paragraph['ref_spans']
                    labels = citations + references
                    for label in sorted(labels, key=lambda x: x['start'], reverse=True):
                        # remove the citation from the text by cutting out the citation
                        text = text[:label['start']] + text[label['end']:]

                    # remove empty brackets that can be left behind by removing the references with regex
                    text = re.sub(r'\s*\[(,\s)*–*\]', '', text)

                    entry["text"] = text

                    # skip if text is too short (character count)
                    if len(entry['text']) < 10:
                        continue

                    counter += 1
                    f.write(json.dumps(entry, ensure_ascii=False))
                    f.write('\n')

        print(f"Created corpus with {counter} entries")


def main():
    """
    Main function that creates a preprocessor and feeds it the data.
    """
    run = Run.get_context()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest='dataset')
    parser.add_argument('--output_file_name', type=str, dest='output_file_name')
    args = parser.parse_args()

    print("Corpus path", args.dataset)
    corpus_location = args.dataset

    workspace = run.experiment.workspace
    metadata = pd.read_csv(f'{corpus_location}/metadata.csv')
    metadata['pmc_json_files'] = [os.path.join(corpus_location, p) for p in metadata['pmc_json_files']]

    out_dir_path = 'processed'

    preprocessor = Preprocessor(output_location=f'{out_dir_path}/{args.output_file_name}.jsonl')

    preprocessor.create_corpus(metadata)

    # Upload to storage
    # Gets the default datastore, this is where we are going to save our new data
    datastore = Datastore.get(workspace, 'workspaceblobstore')

    # Select the directory where we put our processed data and upload it to our datastore
    _ = Dataset.File.upload_directory(out_dir_path, (datastore, "processed"), overwrite=True)

    # Clean up and remove all local data after we have uploaded the data to the datastore
    shutil.rmtree(out_dir_path)


if __name__ == '__main__':
    main()
