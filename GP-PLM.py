import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, DotProduct, RBF
import os
import pandas as pd

def bayes_est(X_test, X_train, t, y, kernel_func_theta, kernel_func_f, s_epsilon):
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
    
    Sigma_Theta_Theta = A_inv+A_inv.dot(B.dot(S_inv.dot(C.dot(A_inv))))
    Sigma_Theta_y = -A_inv.dot(B.dot(S_inv))
    Sigma_y_Theta = Sigma_Theta_y.T
    Sigma_y_y = S_inv
    
    Sigma_y_y_inv = S
    
    M = Sigma_Theta_y.dot(Sigma_y_y_inv)
    mu_Theta_y = M.dot(y)
    
    post_Sigma_Theta = Sigma_Theta_Theta - Sigma_Theta_y.dot(Sigma_y_y_inv.dot(Sigma_y_Theta))
    
    return mu_Theta_y[n:(n+m)], post_Sigma_Theta[n:(n+m), n:(n+m)]

def marginal_likelihood(X, t, y, s_epsilon, kernel_func_theta, kernel_func_f):
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
    (n, d) = X.shape
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

def find_optimal_hyper_RBF(X, t, y, length_theta_list, length_f_list):
    max_value = -np.inf
    optimal_length = (None, None)
    optimal_s = None
    for length_theta in length_theta_list:
        for length_f in length_f_list:
            current_s, current_value = optimize_s(X, t, y, 1./np.var(y), RBF(length_theta), RBF(length_f))
            if current_value > max_value:
                max_value = current_value
                optimal_length = (length_theta, length_f)
                optimal_s = current_s
    return optimal_s, optimal_length

def bayes_est_hyper_RBF(X_test, X_train, t_train, y_train, length_theta_list, length_f_list):
    (n, d) = X_train.shape
    s_epsilon, (length_theta, length_f) = find_optimal_hyper_RBF(X_train, t_train, y_train, length_theta_list, length_f_list)
    posterior_mean, posterior_cov = bayes_est(
        X_test, X_train, t_train, y_train, RBF(length_theta), RBF(length_f), s_epsilon)
    return posterior_mean, posterior_cov

def find_optimal_hyper_Matern(X, t, y, length_theta_list, length_f_list):
    max_value = -np.inf
    optimal_length = (None, None)
    optimal_s = None
    for length_theta in length_f_list:
        for length_f in length_f_list:
            current_s, current_value = optimize_s(X, t, y, 1./np.var(y), Matern(length_theta), Matern(length_f))
            if current_value > max_value:
                max_value = current_value
                optimal_length = (length_theta, length_f)
                optimal_s = current_s
    return optimal_s, optimal_length

def bayes_est_hyper_Matern(X_test, X_train, t_train, y_train, length_theta_list, length_f_list):
    (n, d) = X_train.shape
    s_epsilon, (length_theta, length_f) = find_optimal_hyper_Matern(X_train, t_train, y_train, length_theta_list, length_f_list)
    posterior_mean, posterior_cov = bayes_est(
        X_test, X_train, t_train, y_train, Matern(length_theta), Matern(length_f), s_epsilon)
    return posterior_mean, posterior_cov

working_directory = "./"

length_theta_list = [10**(i/2) for i in range(-6, 7, 1)]
length_f_list = [10**(i/2) for i in range(-6, 7, 1)]

train_file = os.path.join(working_directory, "train.csv")
test_file = os.path.join(working_directory, "test.csv")

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data.drop(['Y', 'T', 'theta'], axis=1).values
X_test = test_data.drop(['Y', 'T', 'theta'], axis=1).values
theta_test = test_data['theta'].values

y_train = train_data['Y'].values
t_train = train_data['T'].values
    
posterior_mean, posterior_cov = bayes_est_hyper_RBF(X_test, X_train, t_train, y_train, length_theta_list, length_f_list)
posterior_var = np.diag(posterior_cov) 

estimation_error = np.mean((theta_test-posterior_mean)**2)
m, _ = X_test.shape
ci_lower = np.zeros(m)
ci_upper = np.zeros(m)
for j in range(m):
    ci_lower[j] = norm.ppf(0.025, loc=posterior_mean[j], scale=np.sqrt(posterior_var[j]))
    ci_upper[j] = norm.ppf(0.975, loc=posterior_mean[j], scale=np.sqrt(posterior_var[j]))
        
coverage = np.mean(np.logical_and(theta_test>ci_lower, theta_test<ci_upper))
ci_length = np.mean(ci_upper-ci_lower)

print(f'mean MSE: {estimation_error}')
print(f'std MSE: {estimation_error}')
print(f'mean coverage: {coverage}')
print(f'std coverage: {coverage}')
print(f'mean ci length: {ci_length}')
print(f'std ci length: {ci_length}')
