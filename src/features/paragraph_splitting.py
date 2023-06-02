'''
Preprocessing of raw GSC public registry
Takes as input:
folder_files_EN: location of all downloaded files
output_filename
years: years for which to preprocess data (list)
'''

from kfp.v2.dsl import InputPath, Dataset, Input
from typing import List, Union


def paragraph_generation(folder_files_EN: Input[Dataset], output_filename: str, years: List[int]):
    """_summary_

    Args:
        folder_files_EN (Input[Dataset]): _description_
        output_filename (str): _description_
        years (List[int]): _description_

    Returns:
        _type_: _description_
    """
    import torch
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    import argparse
    # from azureml.core import Run, Dataset, Datastore
    # from azureml.data.datapath import DataPath
    # from azureml.data import FileDataset
    # from azureml.core import Workspace, Dataset
    from datetime import datetime
    import time
    from tqdm import tqdm
    import re
    import json
    import os
    import glob
    from datetime import datetime

    DATE = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # from google.cloud import storage

    class ParagraphSplitter:
        """_summary_
        """

        def __init__(self, output_filename):
            self.output_dir = f'/gcs/semantic_search_blobstore/preprocessed_corpus'  #'outputs'
            os.makedirs(self.output_dir, exist_ok=True)
            self.output_path = self.output_dir + '/' + output_filename + '.jsonl'

            if os.path.exists(self.output_path):
                print(f'{self.output_path} already exists')
                self.output_path = self.output_dir + '/' + output_filename + '_' + DATE + '.jsonl'
                print(f'Renaming output file to {self.output_path}')

        def find_newest(self, files):
            """_summary_
            if documents have the same id (e.g. ST 474 INIT and ST 474 COR) only keep the most recent one

            Args:
                files (_type_): _description_

            Returns:
                _type_: _description_
            """
            rows = []
            for f in files:
                doc = json.load(open(f))
                dot_cd = doc["dot_cd"]
                doc_id = doc["doc_id"]
                doc_year = doc["doc_year"]
                document_date = doc["document_date"]

                ids = dot_cd + doc_id + doc_year

                row = {'filename': f, 'id': ids, 'date': document_date}
                rows.append(row)
            info_df = pd.DataFrame(rows)
            return info_df.sort_values('date').drop_duplicates('id', keep='last')['filename'].values

        def bad_paragraph(self, s):
            """
            Check if paragraph is 'good', rules of thumb to discard paragraphs with only numbers or other bogus due to pdf to txt conversion
            

            Args:
                s (_type_): _description_

            Returns:
                _type_: _description_
            """
            bad = False
            numbers = sum(c.isdigit() for c in s)
            letters = sum(c.isalpha() for c in s)
            spaces = sum(c.isspace() for c in s)
            others = len(s) - numbers - letters - spaces
            no_go = [
                letters / 3 <= spaces, spaces <= letters / 10, letters < 2 * numbers, spaces < 5, letters < 100, letters
                < 3 * (numbers+others)
            ]  #
            if any(no_go):
                bad = True
            return bad

        def split(self, txt):
            """_summary_

            Args:
                txt (_type_): _description_

            Returns:
                _type_: _description_
            """
            txt = re.sub('\n{4}', '*', txt)

            txt_list = txt.split('\n\n')
            return txt_list

        def write_row(self, out, row):
            """_summary_

            Args:
                out (_type_): _description_
                row (_type_): _description_
            """
            json_record = json.dumps(row, ensure_ascii=False)
            out.write(json_record + '\n')

        def pre_clean(self, txt):
            """
            clean before checking with bad_paragraph()

            Args:
                txt (_type_): _description_

            Returns:
                _type_: _description_
            """
            clean_patterns = [
                '^(.{0,100}EN) ?\*\*\*+', '^(EN.{0,40})\*EN', '^([0-9]{1,3}) ?\*',
                '\*{2,4}([0-9]{1,2}\.?\*[^\*].{0,2000}) ?\.?\*\*\*\*$'
            ]  # patterns to be deleted e.g. footnotes, headers,
            for pattern in clean_patterns:
                txt = re.sub(pattern, '', txt, count=1)

            # substitution to make text readable
            txt = re.sub('\*([0-9]{1,2})\*([A-Z])', '-\\1- \\2', txt)
            txt = re.sub('\*([0-9]{1,2})\*([a-z])', ' \\2', txt)
            txt = re.sub('([A-Z0-9])\*{1}([A-Z0-9])', '\\1\\2', txt, flags=re.IGNORECASE)
            txt = re.sub('([A-Z0-9])\*\*+([A-Z0-9])', '\\1 \\2', txt, flags=re.IGNORECASE)
            txt = re.sub('\*{1}-\*{1}', '-', txt)
            txt = re.sub(' \*+', ' ', txt)
            txt = re.sub('\*+ ', ' ', txt)
            txt = re.sub('\*\*+', ' ', txt)
            txt = re.sub('\*{1}', '', txt)

            return txt

        def post_clean(self, txt):
            """
            clean after paragraph has been selected as 'good'

            Args:
                txt (_type_): _description_

            Returns:
                _type_: _description_
            """
            # bad for making embeddings
            txt = re.sub('.\.\./\.\.\.', '', txt)
            txt = re.sub('__+', '', txt)
            txt = re.sub('\.\.\.+', '.', txt)
            txt = re.sub('--+', '-', txt)
            txt = re.sub('\[\.\.\.\]', '', txt)

            #filter out urls
            link_pattern = '(Available at: | [0-9]{1,3} )?\(?((https|ftp|http):\/\/|www\.|ftp\.)([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?\)?'
            txt = re.sub(link_pattern, '', txt)

            return txt

        def read_json(self, path):
            """_summary_

            Args:
                path (_type_): _description_

            Returns:
                _type_: _description_
            """
            doc = json.load(open(path))
            return doc

        def write_row(self, out, row):
            """_summary_

            Args:
                out (_type_): _description_
                row (_type_): _description_
            """
            json_record = json.dumps(row, ensure_ascii=False)
            out.write(json_record + '\n')

        # Original text is split in passages on delimiter '\n\n' (no real reason except it seems to work good enough and corresponds to page break)
        # fancy futher splitting was not so good, now simply split into even chunks not exceeding max_length tokens
        # provide a small extra margin of eg 20 characters to avoid splitting in the middle of a word and to provide small overlap between neigbouring paragraphs
        def split_passages(self, tokenizer, output_file):
            """_summary_

            Args:
                tokenizer (_type_): _description_
                output_file (_type_): _description_
            """
            files = []
            for year in years:
                files += glob.glob('/gcs/gsc-migrated-blobstore/raw/publicregister/{}/EN/**.json'.format(year))
            print('start filtering files')
            files = self.find_newest(files)
            print('files', files[:5])

            mode = 'a+'
            with open(self.output_path, mode, encoding='utf-8') as out:
                max_length = 250
                extra_margin = 20
                for f in tqdm(files):
                    doc = self.read_json(f)
                    doc_id = doc['immc_identifier']
                    doc_title = doc['search_title']
                    doc_txt = doc['search_text']

                    new_passages = self.split(doc_txt)
                    passage_id = 0

                    for p in new_passages:
                        p = self.pre_clean(p)
                        if not self.bad_paragraph(p):
                            p = self.post_clean(p)
                            tokens = tokenizer.encode(p)
                            if len(tokens) > 10:
                                if len(tokens) > max_length:
                                    chunks = len(tokens) // max_length + 1
                                    len_chunk = len(p) // chunks
                                    for chunk in range(chunks):
                                        # first chunk no extra margin in the beginning
                                        if chunk == 0:
                                            text = p[chunk * len_chunk:(chunk+1) * len_chunk + extra_margin]
                                            try:
                                                text = text.rsplit(' ', 1)[0]
                                            except IndexError as e:
                                                print(e)
                                        # last chunk no extra marging at the end
                                        elif chunk == chunks - 1:
                                            text = p[chunk*len_chunk - extra_margin:(chunk+1) * len_chunk]
                                            try:
                                                text = text.split(' ', 1)[1]
                                            except IndexError as e:
                                                print(e)
                                        # all other chunks
                                        # except IndexError in rare cases where there is no whitespace to split on.
                                        else:
                                            text = p[chunk*len_chunk - extra_margin:(chunk+1) * len_chunk +
                                                     extra_margin]
                                            try:
                                                text = text.split(' ', 1)[1].rsplit(' ', 1)[0]
                                            except IndexError as e:
                                                print(e)
                                        row = {
                                            '_id': doc_id + '_' + str(passage_id),
                                            'doc_id': doc_id,
                                            'title': doc_title,
                                            'paragraph_id': passage_id,
                                            'text': text,
                                            'chunked': True
                                        }
                                        self.write_row(out, row)
                                        passage_id += 1

                                else:
                                    row = {
                                        '_id': doc_id + '_' + str(passage_id),
                                        'doc_id': doc_id,
                                        'title': doc_title,
                                        'paragraph_id': passage_id,
                                        'text': p,
                                        'chunked': False
                                    }
                                    self.write_row(out, row)
                                    passage_id += 1

    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 512
    bi_encoder.tokenizer.truncate_sequences = False
    tokenizer = bi_encoder.tokenizer

    # initialise ParagraphSplitter
    paragraph_splitter = ParagraphSplitter(output_filename)

    # split paragraphs and write jsonl file to azure
    paragraph_splitter.split_passages(tokenizer=tokenizer, output_file=output_filename)


# DEPRACATED
# Used when scripts run from Azure but now on GCP
if __name__ == "__main__":
    # Get parameters
    run = Run.get_context()
    ws = run.experiment.workspace

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_files_EN', type=str, dest='folder_files_EN')
    parser.add_argument(
        '--output_filename', type=str, dest='output_filename', help='File to which corpus.jsonl is written'
    )

    args = parser.parse_args()

    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 512
    bi_encoder.tokenizer.truncate_sequences = False
    tokenizer = bi_encoder.tokenizer

    # initialise ParagraphSplitter
    paragraph_splitter = ParagraphSplitter()

    # split paragraphs and write jsonl file to azure
    paragraph_splitter.split_passages(dir=args.folder_files_EN, tokenizer=tokenizer, output_file=args.output_filename)
    paragraph_splitter.upload_corpus_to_datastore(ws.get_default_datastore(), 'preprocessed_corpus')
