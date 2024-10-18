#!/usr/bin/env python
# coding: utf-8

import pdfplumber
from PyPDF2 import PdfReader


class DataProcess(object):

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = []

    def SlidingWindow(self, sentences, kernel = 512, stride = 1):
        sz = len(sentences)
        cur = ""
        fast = 0
        slow = 0
        while(fast < len(sentences)):
            sentence = sentences[fast]
            if(len(cur + sentence) > kernel  and (cur + sentence) not in self.data):
                self.data.append(cur + sentence + ".")
                cur = cur[len(sentences[slow] + "."):]
                slow = slow + 1
            cur = cur + sentence + "."
            fast = fast + 1


    def ParseAllPage(self, max_seq = 512, min_len = 6):
        all_content = ""
        #i = 0 
             
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if("...................." in text or "contents" in text):
                    continue
                if(len(text) < 1):
                    continue
                if(text.isdigit()):
                    continue
                page_content = page_content + text
                
            if(len(page_content) < min_len):
                continue
            all_content = all_content + page_content
            #i+=1
            #if i > 2:
                #break
        #print("this is the len of the sentence")
        #print(all_content)
        sentences = all_content.split(".")
        #print(len(sentences))
        self.SlidingWindow(sentences, kernel = max_seq)
        #print("sliding window finished")


if __name__ == "__main__":
    dp =  DataProcess(pdf_path = "./data/train_b.pdf")
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    data = dp.data
    out = open("all_text_b.txt", "w")
    for line in data:
        line = line.strip("\n")
        out.write(line)
        out.write("\n")
    out.close()
