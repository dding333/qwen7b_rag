#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('bge-reranker/bge-reranker-large')
print(model_dir)