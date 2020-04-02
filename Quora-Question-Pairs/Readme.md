# Quora-Question-Pairs

This is a Kaggle Challenge. In this we have to identify if the given two questions have same intent or say they are duplicated questions.

[Fork the solution notebook in Colab](https://colab.research.google.com/drive/1pVr3DSfBYlfQA5qhHFkZ7tkhMzkbTi9k)

## Problem
The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same intent.

## Solution
Here I have used a siamese architecture which is built using Bi-LSTM and used Glove 100d as embedding. Loss used is mean_squared_error. Optimizer is AdaDelta. Training and Validaiton metrics of Trained model are given below.
![Performance](screenshots/performance.png)

## Architecture Diagram
![Architecture](screenshots/architecture.png)

## Authors
* Aditya Jain : [Portfolio](https://adityajain.me)