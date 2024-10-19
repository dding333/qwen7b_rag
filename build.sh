#!/bin/bash

# The following commands are used as an example for packaging. Please modify with your own image address when in actual use.

URL=registry.cn-shanghai.aliyuncs.com

IMAGE_NAME=monkey

VERSION=0.1

docker build -t $URL/$IMAGE_NAME:$VERSION .
