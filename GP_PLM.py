import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, DotProduct, RBF

def bayes_est(X_test, X_train, t, y, kernel_func_theta, kernel_func_f, s_epsilon):
    '''
    Input:
    - X_test: Numpy array containing feature values of the test data (shape: [number of test samples, dimension of features])
    - X_train: Numpy array containing feature values of the training data (shape: [number of training samples, dimension of features])
    - t: Numpy array containing the treatment variables of the training data
    - y: Numpy array containing the response variables of the training data
    - kernel_func_theta: Kernel function for theta
    - kernel_func_f: Kernel function for f
    - s_epsilon: Precision of the noise

    Output:
    - mu_theta_y: Mean values of the posterior predictive of the CATE for the test data (shape: [number of test samples,])
    - post_Sigma_theta: Covariance matrix of the posterior predictive of the CATE for the test data (shape: [number of test samples, number of test samples])

    This function performs Bayesian estimation to determine the predictive distribution of the CATE for the test data based on the given training data.
    It uses two kernel functions to model the relationship among feature, treatment, and response, and calculates the predictive mean and variance.
    '''
    (n, d) = X_train.shape
    (m, d) = X_test.shape
    Phi_z = kernel_func_f(X_train)
    Phi_w_nn = kernel_func_theta(X_train)
    Phi_w_mn = kernel_func_theta(X_test, X_train)
    Phi_w_mm = kernel_func_theta(X_test)
    T = np.diag(t)
    
    Phi_w_all = np.zeros((m+n, m+n))
    Phi_w_all[0:n, 0:n] = Phi_w_nn
    Phi_w_all[0:n, n:(m+n)] = Phi_w_mn.T
    Phi_w_all[n:(m+n), 0:n] = Phi_w_mn
    Phi_w_all[n:(m+n), n:(m+n)] = Phi_w_mm
    E_inv = np.zeros((m+2*n, m+2*n))
    E_inv[0:(m+n), 0:(m+n)] = Phi_w_all
    E_inv[(m+n):(m+2*n), (m+n):(m+2*n)] = Phi_z
    
    F = np.zeros((m+2*n, m+2*n))
    F[0:n, 0:n] = s_epsilon*T.dot(T)
    F[0:n, (m+n):(m+2*n)] = s_epsilon*T
    F[(m+n):(m+2*n), 0:n] = s_epsilon*T
    F[(m+n):(m+2*n), (m+n):(m+2*n)] = s_epsilon*np.eye(n)

    '''
    To calculate the inverse of F, we use the matrix formula of Shur complement.
    A, B, C, and D are the block matrices of F.
    '''
    A_inv = E_inv - E_inv.dot(F.dot(np.linalg.inv(np.eye(m+2*n)+E_inv.dot(F))).dot(E_inv))
    B = np.zeros((m+2*n, n))
    B[0:n, 0:n] = -s_epsilon*T
    B[(m+n):(m+2*n), 0:n] = -s_epsilon*np.eye(n)
    C = B.T
    D = s_epsilon*np.eye(n)
    
    S = D - C.dot(A_inv.dot(B))
    try:
        S_inv = np.linalg.inv(S)
    except:
        S_inv = np.linalg.pinv(S)
    
    Sigma_theta_theta = A_inv+A_inv.dot(B.dot(S_inv.dot(C.dot(A_inv))))
    Sigma_theta_y = -A_inv.dot(B.dot(S_inv))
    Sigma_y_theta = Sigma_theta_y.T
    Sigma_y_y = S_inv
    
    Sigma_y_y_inv = S
    
    M = Sigma_theta_y.dot(Sigma_y_y_inv)
    mu_theta_y = M.dot(y)
    
    post_Sigma_theta = Sigma_theta_theta - Sigma_theta_y.dot(Sigma_y_y_inv.dot(Sigma_y_theta))
    
    return mu_theta_y[n:(n+m)], post_Sigma_theta[n:(n+m), n:(n+m)]

def marginal_likelihood(X, t, y, s_epsilon, kernel_func_theta, kernel_func_f):
    '''
    Input: 
    - X: Numpy array containing feature values of the data (shape: [number of samples, dimension of features])
    - t: Numpy array containing the treatment variables of the data
    - y: Numpy array containing the response variables of the data
    - s_epsilon: Scalar representing the precision of the noise
    - kernel_func_theta: Kernel function for theta
    - kernel_func_f: Kernel function for f

    Output:
    - The log of the marginal likelihood of the data given the model parameters, or -∞ if the covariance matrix is singular.

    This function computes the marginal likelihood of the observed data (X, y) under a Bayesian model.
    It calculates the covariance matrix using kernel functions and the provided data, and then evaluates the log of the
    multivariate normal probability density function at y with this covariance.
    '''
    (n, d) = X.shape
    Phi_w = kernel_func_theta(X)
    Phi_z = kernel_func_f(X)
    T = np.diag(t)
    try:
        cov = (1./s_epsilon)*np.eye(n)+T.dot(Phi_w.dot(T))+Phi_z
        return mvn.logpdf(y, cov=cov)
    except:
        return -np.inf

def calc_grad_s(X, t, y, s_epsilon, kernel_func_theta, kernel_func_f):
    '''
    Input:
    - X: Numpy array containing feature values of the data (shape: [number of samples, dimension of features])
    - t: Numpy array containing the treatment variables of the data
    - y: Numpy array containing the response variables of the data
    - s_epsilon: Scalar representing the precision of the noise
    - kernel_func_theta: Kernel function for theta
    - kernel_func_f: Kernel function for f

    Output:
    - grad_s: The gradient of the log marginal likelihood with respect to the noise precision parameter (s_epsilon)

    This function computes the gradient of the log marginal likelihood with respect to the noise precision parameter, s_epsilon.
    '''
    
    (n, d) = X.shape
    Phi_w = kernel_func_theta(X)
    Phi_z = kernel_func_f(X)
        
    T = np.diag(t)
    Sigma = (1./s_epsilon)*np.eye(n)+T.dot(Phi_w.dot(T))+Phi_z
    Sigma_inv = np.linalg.inv(Sigma)

    # gradient w.r.t. s_epsilon
    Sigma_partial_s = -(1./s_epsilon**2)*np.eye(n)
    grad_s = -0.5*np.trace(Sigma_inv.dot(Sigma_partial_s))+0.5*y.dot(Sigma_inv.dot(Sigma_partial_s.dot(Sigma_inv.dot(y))))
    
    return grad_s

def optimize_s(X, t, y, s_epsilon, kernel_func_theta, kernel_func_f, rate=1.0E-1, tol=1.0E-10, max_iter=100):
    '''
    Input:
    - X: Numpy array containing feature values of the data (shape: [number of samples, dimension of features])
    - t: Numpy array containing the treatment variables (outputs) of the data
    - y: Numpy array containing the response variables of the data
    - s_epsilon: Initial value for the noise precision parameter
    - kernel_func_theta: Kernel function for theta
    - kernel_func_f: Kernel function for f
    - rate: Learning rate for the gradient ascent (default: 0.1)
    - tol: Tolerance for the convergence of the likelihood (default: 1.0E-10)
    - max_iter: Maximum number of iterations for the optimization (default: 100)

    Output:
    - out_s_epsilon: Optimized value for the noise precision parameter
    - new_likelihood: The marginal likelihood of the data given the optimized noise precision

    This function optimizes the noise precision parameter (s_epsilon) of a Bayesian model using gradient ascent.
    It iteratively updates the value of s_epsilon based on the gradient of the log marginal likelihood with respect to s_epsilon.
    '''
    out_s_epsilon = s_epsilon
    pre_likelihood = marginal_likelihood(X, t, y, s_epsilon, kernel_func_theta, kernel_func_f)
    for i in range(max_iter):
        grad = calc_grad_s(X, t, y, out_s_epsilon, kernel_func_theta, kernel_func_f)
        out_s_epsilon = out_s_epsilon+rate*grad
        new_likelihood = marginal_likelihood(X, t, y, out_s_epsilon, kernel_func_theta, kernel_func_f)
        if np.abs(new_likelihood-pre_likelihood)<tol:
            break
        pre_likelihood = new_likelihood
    return out_s_epsilon, new_likelihood

def find_optimal_hyper(X, t, y, kernel_func_theta, kernel_func_f, length_theta_list, length_f_list):
    '''
    Input:
    - X: Numpy array containing feature values of the data (shape: [number of samples, dimension of features])
    - t: Numpy array containing the treatment variables (outputs) of the data
    - y: Numpy array containing the response variables of the data
    - kernel_func_theta: Function that generates a kernel for theta, given a length scale
    - kernel_func_f: Function that generates a kernel for f, given a length scale
    - length_theta_list: List of length scale parameters for the theta kernel function
    - length_f_list: List of length scale parameters for the f kernel function

    Output:
    - optimal_s: The optimized noise precision parameter
    - optimal_length: Tuple (length_theta, length_f) representing the optimal length scale parameters for the kernels

    This function finds the optimal hyperparameters for a Bayesian model using a provided kernel function.
    It iterates over combinations of length scale parameters provided in length_theta_list and length_f_list.
    For each combination, it uses the provided kernel functions to create kernels for theta and f, and then optimizes the 
    noise precision parameter (s_epsilon) using the `optimize_s` function. The goal is to find the combination of length scale 
    parameters that yields the highest marginal likelihood, indicating the best fit of the model to the data. 
    The function returns the optimized noise precision and the optimal length scale parameters for the kernels.
    '''
    max_value = -np.inf
    optimal_length = (None, None)
    optimal_s = None
    for length_theta in length_theta_list:
        for length_f in length_f_list:
            current_s, current_value = optimize_s(X, t, y, 1./np.var(y), kernel_func_theta(length_theta), kernel_func_f(length_f))
            if current_value > max_value:
                max_value = current_value
                optimal_length = (length_theta, length_f)
                optimal_s = current_s
    return optimal_s, optimal_length

def bayes_est_hyper(X_test, X_train, t_train, y_train, kernel_func_theta, kernel_func_f, length_theta_list, length_f_list):
    '''
    Input:
    - X_test: Numpy array containing feature values of the test data (shape: [number of test samples, dimension of features])
    - X_train: Numpy array containing feature values of the training data (shape: [number of training samples, dimension of features])
    - t_train: Numpy array containing the treatment variables (outputs) of the training data
    - y_train: Numpy array containing the response variables of the training data
    - length_theta_list: List of length scale parameters for the RBF kernel to model theta
    - length_f_list: List of length scale parameters for the RBF kernel to model f

    Output:
    - posterior_mean: The mean of the posterior predictive distribution of the CATE for the test data
    - posterior_cov: The covariance matrix of the posterior predictive distribution of the CATE for the test data

    This function performs Bayesian estimation of the CATE for a given test dataset using the Radial Basis Function (RBF) kernel.
    First, it finds the optimal hyperparameters (noise precision and length scale parameters for the RBF kernels) 
    for the Bayesian model by using the training data and the specified lists of length scale parameters.
    This is done through the `find_optimal_hyper_RBF` function.
    Then, using these optimal hyperparameters, it computes the posterior mean and covariance of the CATE for the test data using the `bayes_est` function.
    '''
    s_epsilon, (length_theta, length_f) = find_optimal_hyper(X_train, t_train, y_train, kernel_func_theta, kernel_func_f, length_theta_list, length_f_list)
    posterior_mean, posterior_cov = bayes_est(
        X_test, X_train, t_train, y_train, kernel_func_theta(length_theta), kernel_func_f(length_f), s_epsilon)
    return posterior_mean, posterior_cov

def cate_estimator(X_train, t_train, y_train, X_test, *args):
    '''
    Input:
    - X_train: Numpy array containing feature values of the training data
    - t_train: Numpy array containing the treatment variables of the training data
    - y_train: Numpy array containing the response variables of the training data
    - X_test: Numpy array containing feature values of the test data
    - *args: Optional arguments, can be:
        - No additional arguments: uses default RBF kernels for both theta and f.
        - Two additional arguments: uses provided kernel functions for theta and f.
        - Four additional arguments: uses provided kernel functions and lists of length scales for theta and f.

    Output:
    - The posterior mean and covariance of the CATE for the test data, estimated using Bayesian methods.

    This function performs Conditional Average Treatment Effect (CATE) estimation using Bayesian methods. It adapts to different scenarios based on the number of optional arguments provided:
    - If no additional arguments are provided, it uses default RBF kernels for both theta and f, and optimizes the noise precision parameter before performing Bayesian estimation.
    - If two additional arguments are provided, it expects these to be kernel functions for theta and f, and again optimizes the noise precision parameter before performing Bayesian estimation.
    - If four additional arguments are provided, it expects two kernel functions and two lists of length scales for theta and f, and then performs Bayesian estimation with hyperparameter optimization.
    '''
    if len(args) == 0:
        s_epsilon = optimize_s(X_train, t_train, y_train, 1./np.var(y_train), RBF(), RBF())[0]
        return bayes_est(X_test, X_train, t_train, y_train, RBF(), RBF(), s_epsilon)
    
    if len(args) == 2:
        kernel_func_theta, kernel_func_f = args
        s_epsilon = optimize_s(X_train, t_train, y_train, 1./np.var(y_train), kernel_func_theta(), kernel_func_f())[0]
        return bayes_est(X_test, X_train, t_train, y_train, kernel_func_theta(), kernel_func_f(), s_epsilon)
    
    if len(args) == 4:
        kernel_func_theta, kernel_func_f, length_theta_list, length_f_list = args
        return bayes_est_hyper(X_test, X_train, t_train, y_train, kernel_func_theta, kernel_func_f, length_theta_list, length_f_list)