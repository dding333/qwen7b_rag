# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04-cudnn

FROM registry.cn-shanghai.aliyuncs.com/aicar/vllm:base

# If there is a need to install other software
# RUN apt-get update && apt-get install curl

# If there is a need to install additional Python packages
#pip3 install numpy --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# RUN pip install --progress-bar off numpy pandas PyPDF2 langchain jieba rank_bm25 sentence-transformers faiss-gpu modelscope tiktoken transformers_stream_generator accelerate pdfplumber --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Copy the code to the image repository
COPY app /app

# Set the working directory
WORKDIR /app

# Command to run when the container starts
CMD ["bash", "run.sh"]
