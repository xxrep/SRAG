import json
import bz2
import os
import csv
import argparse


def load_data(path, output_path):
    
    count = 0
    retrieval_corpus = {}

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = root + '/' + file
            with bz2.open(file_path, 'r') as fin:
                for line in fin.readlines():
                    data = json.loads(line)
                    row_text = ''
                    json_obj = {}
                    for text in data['text']:
                        row_text = row_text + '' + text
                    
                    json_obj["title"] = data['title']
                    json_obj["text"] = row_text
                    count += 1
                    retrieval_corpus[str(count)] = json_obj
                    print(count)

    with open(output_path, 'w') as json_file:
        json.dump(retrieval_corpus, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Wikipedia paragraphs and save to JSON')
    parser.add_argument('input_dir', type=str, default='./wikipedia-paragraphs', help='Path to the folder containing the Wikipedia paragraphs')
    parser.add_argument('output_path', type=str, default='./wiki_corpus.json', help='Output path')
    args = parser.parse_args()
    load_data(args.input_dir, args.output_path)
