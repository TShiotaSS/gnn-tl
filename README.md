# gnn_tf

This environment provides a platform for using trained NNP graph neural networks as descriptors. 

For more details on the methodology, please refer to the reference section.

# Installing via Docker

We provide a user-friendly container environment with Docker and Jupyter Lab to avoid cumbersome dependencies and environment construction costs.

Execute the following command in the directory where docker-compose.yml is located.
A docker image will be created, referring to the Dockerfile.
```
docker-compose build
```

Run the following command in the directory containing docker-compose.yml.
```
docker-compose up
```

Then, run the following command in some browser. you can connect to the Docker container through Jupyter Lab. The settings are in docker-compose.yml.
```
http://localhost:18080
``` 

# Tutorial 

Please refer to the notebook in the /gnn_tf/tutorial.

# Requirements


# Datasets
The train data and test data are hosted on figshare.
- [qm9nmr training and test dataset](https://figshare.com/account/projects/199786/articles/25484068)
- [12, 40 drugs test dataset](https://figshare.com/account/projects/199786/articles/25484110)

# Reference 

> T. Shiota, K. Ishihara, W. Mizukami, Universal neural network potentials as descriptors: Towards scalable chemical property prediction using quantum and classical computers,[<br>arXiv:2402.18433](https://arxiv.org/abs/2402.18433) [quant-ph]


