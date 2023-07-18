# TensorRT

## Introduction

> TensorRT is a high-performance deep-learning inference engine developed by NVIDIA. It is designed to optimize and accelerate the inference of deep neural networks on NVIDIA GPUs. TensorRT can take trained models from popular deep learning frameworks, such as TensorFlow and PyTorch, and optimize them for deployment on NVIDIA GPUs.

> TensorRT uses a combination of techniques to achieve high performance, including layer fusion, precision calibration, kernel auto-tuning, and dynamic tensor memory management. These techniques enable TensorRT to achieve high throughput and low latency for deep learning inference, making it ideal for applications such as real-time object detection, speech recognition, and natural language processing.

> TensorRT supports a wide range of deep learning operations, including convolution, pooling, activation, and softmax, and it can be used with a variety of deep learning models, including convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformer-based models.

  ![image](https://github.com/A-janjan/tensorrt/assets/62621376/fa93e17d-5f20-41aa-86cf-bd4c80c7a276)


## Goals
- learn how to optimize neural networks with tensorRT [ in Python ]
- learn how to test and compare them
- do some useful projects (real-time or offline) 

## Prerequisite
- some knowledge of Python programming
- be familiar with Pytorch and TensorFlow
- be familiar with artificial neural networks
- having a computer with installed Cuda

## Tested devices
- Tesla T4 (google Colab) [ remote ]
- NVIDIA Jetson AGX Xavier [ local ]

---------------------------------------------------------

>> Current status: working on Pytorch optimization on Jetson AGX Xavier and compare results
