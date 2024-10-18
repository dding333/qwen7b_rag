#!/usr/bin/env python
# coding: utf-8

import json
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
import time
import re

from vllm_model import ChatLLM
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_ALLOW_DEPRECATED_BEAM_SEARCH"] = "1"



# merged the retrieved information using faiss and bm25
def get_emb_bm25_merge(faiss_context, bm25_context, query):
    max_length = 2500
    emb_ans = ""
    cnt = 0
    for doc, score in faiss_context:
        cnt =cnt + 1
        if(cnt>6):
            break
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if(len(bm25_ans + doc.page_content) > max_length):
            break
        bm25_ans = bm25_ans + doc.page_content
        if(cnt > 6):
            break

    prompt_template = """Based on the following known information, provide a concise and professional response to the user's question.
If no answer can be derived from it, say "no answer." Fabrications are not allowed in the response and the known content is from Tesla user manual:
                                1: {emb_ans}
                                2: {bm25_ans}
                                question:
                                {question}""".format(emb_ans=emb_ans, bm25_ans = bm25_ans, question = query)
    return prompt_template


def get_rerank(emb_ans, query):

    prompt_template = """Based on the following known information, provide a concise and professional response to the user's question.
If no answer can be derived from it, say "no answer." Fabrications are not allowed in the response and the known content is from Tesla user manual:
                                1: {emb_ans}
                                question:
                                {question}""".format(emb_ans=emb_ans, question = query)
    return prompt_template



def reRank(rerank, top_k, query, bm25_ans, faiss_ans):
    items = []
    max_length = 4000
    for doc, score in faiss_ans:
        items.append(doc)
    items.extend(bm25_ans)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[:top_k]
    # docs_sort = sorted(rerank_ans, key = lambda x:x.metadata["id"])
    emb_ans = ""
    for doc in rerank_ans:
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans

if __name__ == "__main__":

    start = time.time()

    base = "."
    qwen7 = "/scratch365/dding3/p1/pre_train_model/Qwen-7B-Chat/Qwen-7B-Chat/"
    dense_model = "mixedbread-ai/mxbai-embed-large-v1"
    bge_reranker_large = base + "/pre_train_model/bge-reranker-large"
    
    # base model qwen7
    llm = ChatLLM(qwen7)
    print("llm qwen load ok")

    # pdf sliding window parsing 
    dp =  DataProcess(pdf_path = base + "/data/train_b.pdf")
    dp.ParseAllPage(max_seq = 256)
    #dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    data = dp.data
    print("data load ok")

    # Faiss retrieval
    
    faissretriever = FaissRetriever(dense_model , data)
    print("faissretriever load ok")

    # BM25 retrieval 
    bm25 = BM25(data)
    print("bm25 load ok")

    #breakpoint()

    # Initialize the reRank model 
    #rerank = reRankLLM(bge_reranker_large)
    #print("rerank model load ok")

    # load the all the questions
    with open(base + "/data/input.json", "r") as f:
        jdata = json.loads(f.read())
        print(len(jdata))
        max_length = 4000
        for idx, line in enumerate(jdata):
            query = line["question"]

            # faiss topk
            faiss_context = faissretriever.GetTopK(query, 5)
            faiss_min_score = 0.0
            if(len(faiss_context) > 0):
                faiss_min_score = faiss_context[0][1]
            cnt = 0
            emb_ans = ""
            for doc, score in faiss_context:
                cnt =cnt + 1
                # truncate the length of faiss retrieval 
                if(len(emb_ans + doc.page_content) > max_length):
                    break
                emb_ans = emb_ans + doc.page_content
                # set top as 6 
                if(cnt>6):
                    break

            # bm2.5 topk 
            bm25_context = bm25.GetBM25TopK(query, 5)
            bm25_ans = ""
            cnt = 0
            for doc in bm25_context:
                cnt = cnt + 1
                if(len(bm25_ans + doc.page_content) > max_length):
                    break
                bm25_ans = bm25_ans + doc.page_content
                if(cnt > 6):
                    break

            # merge the faiss context and bm25 context into the prompt 
            emb_bm25_merge_inputs = get_emb_bm25_merge(faiss_context, bm25_context, query)

            # bm25 context to prompt 
            bm25_inputs = get_rerank(bm25_ans, query)

            # faiss context to prompt 
            emb_inputs = get_rerank(emb_ans, query)

            # rerank the bm25 and faiss context 
            #rerank_ans = reRank(rerank, 6, query, bm25_context, faiss_context)
            # reranked context to prompt 
            #rerank_inputs = get_rerank(rerank_ans, query)

            batch_input = []
            batch_input.append(emb_bm25_merge_inputs)
            batch_input.append(bm25_inputs)
            batch_input.append(emb_inputs)
            #batch_input.append(rerank_inputs)
            # batch inference 
            batch_output = llm.infer(batch_input)
            line["answer_1"] = batch_output[0] # result with merged context from bm 25 and faiss
            line["answer_2"] = batch_output[1] # result with context from bm 25
            line["answer_3"] = batch_output[2] # result with merged context from faiss
            #line["answer_4"] = batch_output[3] # result with rerank 
            #line["answer_5"] = emb_ans
            #line["answer_6"] = bm25_ans
            #line["answer_7"] = rerank_ans
            # if the faiss retrieval and query hava a distance > 500: output no answer 
            #if(faiss_min_score >500):
                #line["answer_5"] = "no answer"
            #else:
                #line["answer_5"] = str(faiss_min_score)

        # save the result to the ouptput.json file
        json.dump(jdata, open(base + "/data/new_output.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        end = time.time()
        print("cost time: " + str(int(end-start)/60))