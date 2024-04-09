# gnn_tf

This environment provides a platform for using trained NNP graph neural networks as descriptors. 

As a typical example, we provide a predictive model of NMR chemical shifts using the universal M3GNet GNN as a descriptor.

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

Please refer to the notebook in the [/gnn_tf/tutorial](https://github.com/TShiotaSS/gnn_tf/blob/main/tutorial/1_generate_m3gnet_descriptor/m3gnet_gnntf_generator.ipynb).

# Requirements
- Python >= 3.9

# Datasets
The train data and test data are hosted on figshare.
- [qm9nmr training and test dataset](https://figshare.com/articles/dataset/M3GNet_descriptors_for_organic_molecules_in_QM9NMR_dataset/25484068)
# Reference 

> T. Shiota, K. Ishihara, W. Mizukami, Universal neural network potentials as descriptors: Towards scalable chemical property prediction using quantum and classical computers,[<br>arXiv:2402.18433](https://arxiv.org/abs/2402.18433) [quant-ph]

> Chen, C., Ong, S.P. A universal graph deep learning interatomic potential for the periodic table. Nat Comput Sci 2, 718–728 (2022). https://doi.org/10.1038/s43588-022-00349-3.

> scikit-qulacs: https://github.com/Qulacs-Osaka/scikit-qulacs 
