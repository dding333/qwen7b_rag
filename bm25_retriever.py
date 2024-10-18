#!/usr/bin/env python
# coding: utf-8


from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from pdf_parse import DataProcess
import nltk
from nltk.tokenize import word_tokenize

class BM25(object):

    # Initialize by tokenizing the documents and creating an index for BM25 
    def __init__(self, documents):

        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip("\n").strip()
            if(len(line)<5):
                continue
            tokens = " ".join(word_tokenize(line))
            # docs.append(Document(page_content=tokens, metadata={"id": idx, "cate":words[1],"pageid":words[2]}))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            # full_docs.append(Document(page_content=words[0], metadata={"id": idx, "cate":words[1], "pageid":words[2]}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        
        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_bm25()


    # initialize bm25
    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    # get top k function
    def GetBM25TopK(self, query, topk):
        self.retriever.k = topk
        query = " ".join(word_tokenize(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans

if __name__ == "__main__":

    # bm2.5
    dp =  DataProcess(pdf_path = "./data/train_b.pdf")
    dp.ParseAllPage(max_seq = 256)
    #dp.ParseAllPage(max_seq = 512)
    data = dp.data
    print("this is the len of the data")
    print(len(dp.data))
    print("this is len of the data[0]")
    print(len(data[0]))
    bm25 = BM25(data)
    res = bm25.GetBM25TopK("tesla", 6)
    print("this is the res[0]")
    print(res[0])
