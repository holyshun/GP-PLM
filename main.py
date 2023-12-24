import pandas as pd
import os
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF
from GP_PLM import cate_estimator

working_directory = "./"

length_theta_list = [10**(i/2) for i in range(-1, 2, 1)]
length_f_list = [10**(i/2) for i in range(-1, 2, 1)]

train_file = os.path.join(working_directory, "train.csv")
test_file = os.path.join(working_directory, "test.csv")

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data.drop(['Y', 'T', 'theta'], axis=1).values
X_test = test_data.drop(['Y', 'T', 'theta'], axis=1).values
theta_test = test_data['theta'].values

y_train = train_data['Y'].values
t_train = train_data['T'].values

posterior_mean, posterior_cov = cate_estimator(X_train, t_train, y_train, X_test, RBF, RBF, length_theta_list, length_f_list)
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
print(f'mean credible interval coverage: {coverage}')
print(f'mean credible interval length: {ci_length}')