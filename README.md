# gnn_tf

This library provides a platform for using trained NNP graph neural networks as descriptors. 

For more details on the methodology, please refer to the reference section.

# Installing via Docker

Execute the following command in the directory where docker-compose.yml is located.
A docker image will be created, referring to the Dockerfile.
```
docker-compose build
``` docker-compose build

Run the following command in the directory containing docker-compose.yml.
```
docker-compose up
```

# Example 

Please refer to the notebook in the /gnn_tf/example.

# Requirements


# Datasets
The train data and test data are hosted on figshare.
- [qm9nmr training and test dataset](https://figshare.com/account/projects/199786/articles/25484068)
- [12, 40 drugs test dataset](https://figshare.com/account/projects/199786/articles/25484110)

# Reference 

> T. Shiota, K. Ishihara, W. Mizukami, Universal neural network potentials as descriptors: Towards scalable chemical property prediction using quantum and classical computers,[<br>arXiv:2402.18433](https://arxiv.org/abs/2402.18433) [quant-ph]


