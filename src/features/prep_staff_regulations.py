'''
Split staff regulations, done locally as only 150 pages, no need for fancy pipelines
staff regulations as downloaded in October 2022 from https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:01962R0031-20140501&from=EN
'''

from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import re
import json
import unicodedata
from bs4 import BeautifulSoup
import codecs
from google.oauth2 import service_account
from google.cloud import storage


class StaffRegulationSplitter:
    """_summary_
    """

    def __init__(self):
        self.output_dir = 'processed_staff_regulations'
        try:
            os.makedirs(self.output_dir, exist_ok=False)
        except FileExistsError as e:
            #rint('output directory already exists and it is assumed processing is already done. To redo the processing delete the folder and its contents')
            raise FileExistsError(
                'output directory already exists and it is assumed processing is already done. To redo the processing delete the folder and its contents.'
            )

    def bad_paragraph(self, s):
        """_summary_

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
        no_go = [letters < 3 * spaces, letters < 2 * numbers, spaces < 5, letters < 1.3 * others, letters == 0]
        if any(no_go):
            bad = True
        return bad

    def split_chapters(self, txt):
        """_summary_

        Args:
            txt (_type_): _description_

        Returns:
            _type_: _description_
        """
        txt = unicodedata.normalize('NFKD', txt)
        pats = ['\n►[A-Z][0-9]{0,3}\n', '►[A-Z][0-9]{0,3}', '◄[A-Z][0-9]{0,3}', '▼[A-Z][0-9]{0,3}', '▼B', '►', '◄', '▼']
        for pat in pats:
            txt = re.sub(pat, '', txt)
        regex = 'TITLE \n{0,3}.{0,20}I+.?\n?|TITLE \n{0,3}.{0,20}IV.?\n?|TITLE \n{0,3}.{0,20}V+.?\n?|TITLE \n{0,3}.{0,20}VI+.?\n?|TITLE \n{0,3}.{0,20}IX.?\n?|TITLE \n{0,3}.{0,20}XI+.?\n?'
        txt = re.split(regex, txt)

        txt_list = txt  #txt.split('\n\n')
        return txt_list

    def write_row(self, out, row):
        """_summary_

        Args:
            out (_type_): _description_
            row (_type_): _description_
        """
        json_record = json.dumps(row, ensure_ascii=False)
        out.write(json_record + '\n')

    def split_staff_regulations(self, doc, tokenizer, output_file):
        """_summary_

        Args:
            doc (_type_): _description_
            tokenizer (_type_): _description_
            output_file (_type_): _description_
        """
        mode = 'a+'
        max_length = 510
        extra_margin = 20
        output_path = self.output_dir + '/' + output_file + '.jsonl'
        with open(output_path, mode, encoding='utf-8') as out:
            for t in doc:
                passage_id = 0
                d = re.split('^ {0,2}\n{0,4}(.{4,100})\n+', t, maxsplit=1)
                if len(d) < 2:
                    print(d)
                title = d[1]
                txt = d[2]
                print(title)
                articles = re.split('(Article +[0-9]{0,3}[a-z]?) {0,2}\n+', txt)
                for i in range(1, len(articles) - 1):
                    # after splitting txt on title, uneven indices contain title while even indices contain text
                    if i % 2 != 0:
                        doc_id = articles[i]
                    if i % 2 == 0:
                        article = re.sub('\n+CHAPTER.{0,200}$|\n+Section.{0,200}$', '', articles[i], flags=re.DOTALL)
                        article = re.sub(
                            '\n+ANNEX {0,3}(?:XIII\.1|IV|IX|VI+|V|XI+|X|I+).*', '', article, flags=re.DOTALL
                        )
                        article = re.sub('\n+', '', article)
                        article = re.sub(' +', ' ', article)
                        article = re.sub('--+', '', article)

                        if not self.bad_paragraph(article):
                            tokens = tokenizer.encode(article)
                            if len(tokens) > max_length:
                                chunks = len(tokens) // max_length + 1
                                len_chunk = len(article) // chunks
                                for chunk in range(chunks):
                                    if chunk == 0:
                                        text = article[chunk * len_chunk:(chunk+1) * len_chunk + extra_margin]
                                        text = text.rsplit(' ', 1)[0]
                                    elif chunk == chunks - 1:
                                        text = article[chunk*len_chunk - extra_margin:(chunk+1) * len_chunk]
                                        text = text.split(' ', 1)[1]
                                    else:
                                        text = article[chunk*len_chunk - extra_margin:(chunk+1) * len_chunk +
                                                       extra_margin]
                                        text = text.split(' ', 1)[1].rsplit(' ', 1)[0]
                                    row = {
                                        '_id': title + ' ' + str(passage_id),
                                        'doc_id': doc_id,
                                        'title': title,
                                        'paragraph_id': passage_id,
                                        'text': text,
                                        'chunked': True
                                    }
                                    self.write_row(out, row)
                                    passage_id += 1

                            else:
                                row = {
                                    '_id': title + ' ' + str(passage_id),
                                    'doc_id': doc_id,
                                    'title': title,
                                    'paragraph_id': passage_id,
                                    'text': article,
                                    'chunked': False
                                }
                                self.write_row(out, row)
                                passage_id += 1


if __name__ == "__main__":

    # Load tokenizer
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 512  #Truncate long passages
    bi_encoder.tokenizer.truncate_sequences = False
    tokenizer = bi_encoder.tokenizer

    # open staffregulations downloaded from https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:01962R0031-20140501&from=EN
    f = codecs.open('data/raw/staff_regulations.html', 'r', 'utf-8')
    staffreg_html = BeautifulSoup(f.read())
    staffreg_doc = staffreg_html.get_text()

    # process staff regulations by splitting to articles
    processor = StaffRegulationSplitter()
    chapters = processor.split_chapters(staffreg_doc)

    eu_officials_doc = chapters[11:21]
    other_servants_doc = chapters[29:]

    processor.split_staff_regulations(eu_officials_doc, tokenizer, 'stafreg_eu_officials')
    processor.split_staff_regulations(other_servants_doc, tokenizer, 'stafreg_other_servants')

    # upload results to gcp
    credentials = service_account.Credentials.from_service_account_file('.cloud/.gcp/GCP_SERVICE_ACCOUNT.json')
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket('semantic_search_blobstore')

    for f in os.listdir(processor.output_dir):
        dest = f'preprocessed_corpus/staff_regulations/{f}'
        blob = bucket.blob(dest)
        source = os.path.join(processor.output_dir, f)
        blob.upload_from_filename(source)
