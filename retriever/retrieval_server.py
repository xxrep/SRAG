import pprint
import json
import copy
import os
import numpy as np
from tqdm import tqdm
import time
import argparse
import pathlib

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense import FlatIPFaissSearch as FIFS
from beir.reranking.models import CrossEncoder
from beir.retrieval import models
from beir.reranking import Rerank

import src_con.contriever
from src_con.dense_model import DenseEncoderModel
from fastapi import FastAPI, Request
import uvicorn, json, datetime

app = FastAPI()
# torch.cuda.set_device(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


@app.post("/")
async def retrieve_info(request: Request):
    global retriever
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    questions = json_post_list.get('questions')
    response = retriever.run(questions)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    print('----------------------------')
    print("[" + time + "]")
    print("question_list: {}".format(questions))
    print("search_result: {}".format(response))
    print('----------------------------')
    return answer

class Retriever:
    def __init__(self, data_name, topk=5, corpus_path='./corpus/wiki.json', retrieval_method='bm25', elastic_search_server='localhost:9200', index_init=False):
        self.data_name = data_name
        self.topk = topk
        print("=====> Loading Pool of Documents")
        self.retrieval_corpus = json.load(open(corpus_path))
        self.retrieval_method = retrieval_method
        if retrieval_method == 'bm25':
            self.model = BM25(hostname=elastic_search_server, index_name=data_name + "_bm25", initialize=index_init)
            self.retriever = EvaluateRetrieval(self.model)
        elif retrieval_method == 'contriever':
            model, tokenizer, _ = src_con.contriever.load_retriever('facebook/contriever-msmarco')
            model = model.cuda()
            model.eval()
            query_encoder = model
            doc_encoder = model
            self.model = FIFS(DenseEncoderModel(query_encoder=query_encoder, doc_encoder=doc_encoder, tokenizer=tokenizer),batch_size=256)
            self.load_index()
            self.retriever = EvaluateRetrieval(self.model, score_function="dot")
        elif retrieval_method == 'dpr':
            self.model = FIFS(models.SentenceBERT(("facebook-dpr-question_encoder-multiset-base",
                                  "facebook-dpr-ctx_encoder-multiset-base",
                                  " [SEP] "), batch_size=256))
            self.load_index()
            self.retriever = EvaluateRetrieval(self.model, score_function="dot")
        else:
            raise ValueError("Wrong retrieval method is inserted.")
    
    def load_index(self):
        prefix = "my-index"
        ext = "flat"
        index_path = "./raw_data/{}/{}/faiss-index".format(self.data_name, self.retrieval_method)
        if os.path.exists(os.path.join(index_path, "{}.{}.faiss".format(prefix, ext))):
            print("=====> Loading Indexes")
            self.model.load(input_dir=index_path, prefix=prefix, ext=ext)
    
    def save_index(self):
        prefix = "my-index"
        ext = "flat"
        index_path = "./raw_data/{}/{}/faiss-index".format(self.data_name, self.retrieval_method)
        os.makedirs(index_path, exist_ok=True)
        if not os.path.exists(os.path.join(index_path, "{}.{}.faiss".format(prefix, ext))):
            self.model.save(output_dir=index_path, prefix=prefix, ext=ext)

    def run(self, questions):
        retrieval_queries = {}
        for i in range(len(questions)):
            question = questions[i]
            qa_id = str(self.data_name) + "_" + str(i)
            retrieval_queries[qa_id] = question
        retrieval_scores = self.retriever.retrieve(self.retrieval_corpus, retrieval_queries)
        
        if self.retrieval_method != 'bm25':
            self.save_index()

        sorted_idxs = []
        sorted_scores = []
        for i in range(len(retrieval_scores)):
            scores_i = np.array(list(retrieval_scores['{}_{}'.format(self.data_name, i)].values()))
            sorted_idx = np.argsort(scores_i)[::-1]
            keys = list(retrieval_scores['{}_{}'.format(self.data_name, i)].keys())

            sorted_idxs_i = []
            sorted_scores_i = []
            for j in range(min(len(scores_i), self.topk)):
                sorted_idxs_i.append(int(keys[sorted_idx[j]]))
                sorted_scores_i.append(scores_i[sorted_idx[j]])

            sorted_idxs.append(sorted_idxs_i)
            sorted_scores.append(sorted_scores_i)

        res = []
        for i in range(len(questions)):
            new_item = {}
            new_item['question'] = questions[i]
            # new_item['answer'] = answers[i]
            ctxs = []
            for j in range(len(sorted_idxs[i])):
                ctx = {}
                ctx['id'] = sorted_idxs[i][j]
                # ctx['title'] = titles[sorted_idxs[i][j]]
                ctx['title'] = self.retrieval_corpus[str(sorted_idxs[i][j])]["title"]
                ctx['text'] = self.retrieval_corpus[str(sorted_idxs[i][j])]["text"]
                ctx['score'] = sorted_scores[i][j]
                ctxs.append(ctx)
            new_item['contexts'] = ctxs
            res.append(new_item)
        
        return res
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval server with different settings")
    
    parser.add_argument('--data_name', type=str, default='hotpotqa')
    parser.add_argument('--corpus_path', type=str, default='./corpus/wiki.json', help='Path to wiki corpus')
    parser.add_argument('--retrieval_method', type=str, default='bm25', help='bm25 / contriever / dpr')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--port', type=int, default=8001)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    retriever = Retriever(args.data_name, topk=args.topk, corpus_path=args.corpus_path, retrieval_method=args.retrieval_method)
    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)
    