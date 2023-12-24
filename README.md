# Bayesian CATE estimation with Gaussian-Process-Based Partially Linear Model

This repository contains the implementation of the methods described in the paper  
Shunsuke Horii, Yoichi Chikahara; "Uncertainty Quantification in Heterogeneous Treatment Effect Estimation with Gaussian-Process-Based Partially Linear Model" to appear in AAAI 2024.  

## Overview

This project implements the Bayesian estimator of Conditional Average Treatment Effect (CATE) with Gaussian-Process-Based Partially Linear Model, as presented in the aforementioned paper.

## Requirements

Following libraries are required:
- numpy
- scipy
- pandas
- scikit_learn

To install requirements:

```setup  
git clone https://github.com/holyshun/GP-PLM.git
cd GP-PLM
pip install -r requirements.txt
```

## Usage

The *cate_estimator* function in this repository is designed to estimate CATE using Bayesian methods with varying levels of customization based on the provided arguments. Below are examples of how to use this function:

### Basic Usage with Default RBF Kernels
```
from GP_PLM import cate_estimator
import numpy as np

# Sample data preparation
X_train = np.array([[...]])  # Your training feature data
t_train = np.array([...])    # Your training treatment data
y_train = np.array([...])    # Your training response data
X_test = np.array([[...]])   # Your test feature data

# Using the function with default RBF kernels
posterior_mean, posterior_cov = cate_estimator(X_train, t_train, y_train, X_test)
```

### Usage with specified Kernel Functions
```
from sklearn.gaussian_process.kernels import Matern

# Using the function with Matérn kernel functions
posterior_mean, posterior_cov = cate_estimator(X_train, t_train, y_train, X_test, Matern, Matern)
```

### Usage with Hyperparameter Optimization
```
# Define lists of length scales for hyperparameter optimization
length_theta_list = [...]
length_f_list = [...]

# Using the function with hyperparameter optimization and Matérn kernel
posterior_mean, posterior_cov = cate_estimator(X_train, t_train, y_train, X_test, Matern, Matern, length_theta_list, length_f_list)
```

## License

This repository is licensed under "LICENSE"
