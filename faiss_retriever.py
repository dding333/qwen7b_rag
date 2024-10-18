#!/usr/bin/env python
# coding: utf-8


from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
import torch
import faiss


from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from typing import Dict


# Custom embedding function based on your provided embedding code
def pooling(outputs, inputs, strategy='cls'):
    if strategy == 'cls':
        outputs = outputs[:, 0]  # CLS token pooling
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
    else:
        raise NotImplementedError(f"Pooling strategy {strategy} is not implemented")
    return outputs.detach().cpu().numpy()


class FaissRetriever(object):
    # Initialize document chunk index and then insert into FAISS
    def __init__(self, model_path, data, batch_size=1):
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).cuda()
        self.model.gradient_checkpointing_enable()

        # Process documents in batches
        docs = []
        all_embeddings = []
        for idx in range(0, len(data), batch_size):
            batch = data[idx:idx + batch_size]
            batch_docs = []
            batch_texts = []
            for line in batch:
                line = line.strip("\n").strip()
                words = line.split("\t")
                batch_docs.append(Document(page_content=words[0], metadata={"id": idx}))
                batch_texts.append(words[0])

            docs.extend(batch_docs)

            # Tokenize the batch
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

            # Move token indices to GPU
            for k, v in inputs.items():
                inputs[k] = v.cuda()

            # Get the model's last hidden state and apply pooling
            outputs = self.model(**inputs).last_hidden_state
            doc_embeddings = pooling(outputs, inputs, 'cls')

            all_embeddings.append(doc_embeddings)

        all_embeddings = np.vstack(all_embeddings)  # Stack all batches together

        # Create FAISS index with the generated embeddings
        embedding_dim = all_embeddings.shape[1]  # Assuming the embeddings are 2D: (num_docs, embedding_dim)
        self.index = faiss.IndexFlatL2(embedding_dim)  # Create an FAISS index

        # Add the embeddings to the FAISS index
        self.index.add(all_embeddings)

        # Save the document metadata
        self.docs = docs

        # Free up memory
        torch.cuda.empty_cache()

    # Retrieve top-K highest scoring document chunks
    def GetTopK(self, query, k):
        # Ensure k is an integer
        k = int(k)

        # Generate query embedding using the same method
        inputs = self.tokenizer([query], padding=True, return_tensors='pt')
        for key, value in inputs.items():
            inputs[key] = value.cuda()

        outputs = self.model(**inputs).last_hidden_state
        query_embedding = pooling(outputs, inputs, 'cls')

        # FAISS expects input to be 2D (n_queries, embedding_dim), ensure the correct shape
        query_embedding = np.expand_dims(query_embedding, axis=0) if query_embedding.ndim == 1 else query_embedding

        # Perform similarity search in FAISS
        distances, indices = self.index.search(query_embedding, k)

        # Retrieve the top-K documents based on the indices
        top_k_docs = [(self.docs[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
        return top_k_docs

    # Return FAISS index object
    def GetvectorStore(self):
        return self.index


if __name__ == "__main__":
    torch.cuda.empty_cache()
    base = "."
    model_name = "mixedbread-ai/mxbai-embed-large-v1"  # New embedding model
    dp = DataProcess(pdf_path=base + "/data/train_b.pdf")

    dp.ParseAllPage(max_seq = 256)
    print(len(dp.data))

    data = dp.data
    print(data[0])

    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("what should I do when the battery is dead", int(6))
    print(faiss_ans)


