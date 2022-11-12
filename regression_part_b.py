import torch
import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
import torch
import scipy.stats as st
import sklearn.linear_model as lm


# Load the Glass Identification Data Set csv data using the Pandas library
filename = 'dataset/glass.data'
glass_df = pd.read_csv(filename, names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"])

X = glass_df[["RI", "Na", "Mg", "Al", "Si", "K", "Ba", "Fe"]].to_numpy(dtype=np.float32)
y = glass_df[["Ca"]].to_numpy(dtype=np.float32)

N = np.shape(X)[0] # Number of observations
M = np.shape(X)[1] # Number of attributes

# Since we're training a neural network for regression, we use a 
# mean square error loss
loss_fn = torch.nn.MSELoss()
max_iter = 10000
device = torch.device('cuda:0')



def train_neural_net(net, loss_fn, X, y,
                     n_replicates=3, max_iter = 10000, tolerance=1e-6):
    # Specify maximum number of iterations for training
    logging_frequency = 1000 # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        # print('\n\t\tReplicate: {}/{}'.format(r+1, n_replicates))         
        optimizer = torch.optim.Adam(net.parameters())
        
        # Train the network while displaying and storing the loss
        # print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X) # forward pass, predict labels on training set
            loss = loss_fn(y_est.flatten(), y.flatten()) # determine loss
            loss_value = loss.cpu().data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # record loss for later display
            
            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value
            
            # display loss with some frequency:
            # if (i != 0) & ((i+1) % logging_frequency == 0):
                # print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                # print(print_str)
            # do backpropagation of loss and optimize weights 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            
        # # # display final loss
        # print('\t\t\tFinal loss:')
        # print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        # print(print_str)
        
        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve


class Net(torch.nn.Module):

    def __init__(self, n_hidden_units):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                    )
        
        
    def forward(self, x):
        return self.model.forward(x)


def inner_kfold_ANN(X,y,hidden_units,cvf=10):
    CV = model_selection.KFold(cvf, shuffle=True, random_state=42)
    train_error = np.empty((cvf,len(hidden_units)))
    test_error = np.empty((cvf,len(hidden_units)))
    f = 0
    y = y.squeeze()
    k=0
    for train_index, test_index in CV.split(X,y):
        print("\n\t Inner fold:", k)
        k+=1
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
    
        for h in range(0,len(hidden_units)):
            # Construct a network with the current number of hidden units
            n_hidden_units = hidden_units[h]
            model = Net(n_hidden_units)
            model = model.to(device)
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=torch.Tensor(X_train).to(device),
                                                       y=torch.Tensor(y_train).to(device),
                                                       n_replicates=3,
                                                       max_iter=max_iter)

            # Determine estimated calcium percentage for training and test set
            y_train_est = net(torch.Tensor(X_train).to(device)).cpu().data.numpy().flatten()  # prediction of network
            y_train = y_train

            y_test_est = net(torch.Tensor(X_test).to(device)).cpu().data.numpy().flatten()  # prediction of network
            y_test = y_test

            # Evaluate training and test performance
            train_error[f,h] = np.power(y_train-y_train_est, 2).mean(axis=0)
            test_error[f,h] = np.power(y_test-y_test_est, 2).mean(axis=0)
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_hidden_units = hidden_units[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_hidden_units = np.mean(train_error,axis=0)
    test_err_vs_hidden_units = np.mean(test_error,axis=0)
    
    return opt_val_err, opt_hidden_units, train_err_vs_hidden_units, test_err_vs_hidden_units

def rlr_validate(X,y,lambdas,cvf=10):
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda
        

K_outter = 10
K_inner = 10
CV = model_selection.KFold(K_outter, shuffle=True, random_state=42)

# Initialize variables
Error_train = np.empty((K_outter,1))
Error_test = np.empty((K_outter,1))

Error_train_ANN = np.empty((K_outter,1))
Error_test_ANN = np.empty((K_outter,1))

Error_train_rlr = np.empty((K_outter,1))
Error_test_rlr = np.empty((K_outter,1))

Error_train_baseline = np.empty((K_outter,1))
Error_test_baseline = np.empty((K_outter,1))

Optimal_h_history = np.empty((K_outter,1))
Optimal_lambda_history = np.empty((K_outter,1))

w_rlr = np.empty((M,K_outter))

yhat_ANN = []
yhat_lr = []
yhat_baseline = []
y_true = []

# Numbers of hidden units
hidden_units = [1, 16, 64, 256]#, 128, 256]
# Values of lambda
lambdas = np.power(10.,range(-5,9))

k = 0
for train_index, test_index in CV.split(X,y):
    print("\n Outter fold:", k)
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]   
    
    opt_val_err, opt_h, train_err_vs_lambda, test_err_vs_lambda = inner_kfold_ANN(X_train, y_train, hidden_units, K_inner)
    # # Build the optimal ANN, on the entire training set
    n_hidden_units = opt_h
    model = Net(n_hidden_units)
    model = model.to(device)

    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=torch.Tensor(X_train).to(device),
                                                       y=torch.Tensor(y_train).to(device),
                                                       n_replicates=3,
                                                       max_iter=max_iter)
    
    # Determine estimated calcium percentage for training and test set
    y_train_est_ANN = net(torch.Tensor(X_train).to(device)).cpu().data.numpy()  # prediction of network
    y_test_est_ANN = net(torch.Tensor(X_test).to(device)).cpu().data.numpy()  # prediction of network

    # Build the linear model regression 
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K_inner)
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()

    y_train_est_lr = X_train @ w_rlr[:,k]
    y_train_est_lr = y_train_est_lr.reshape(y_train_est_lr.shape[0],1)

    y_test_est_lr = X_test @ w_rlr[:,k]
    y_test_est_lr = y_test_est_lr.reshape(y_test_est_lr.shape[0],1)

    # Compute mean squared error for ANN
    Error_train_ANN[k] = np.square(y_train-y_train_est_ANN).sum(axis=0)/y_train.shape[0]
    Error_test_ANN[k] = np.square(y_test-y_test_est_ANN).sum(axis=0)/y_test.shape[0]

    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-y_train_est_lr).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-y_test_est_lr).sum(axis=0)/y_test.shape[0]

    # Compute mean squared error without using the input data at all
    Error_train_baseline[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_baseline[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    yhat_ANN.append(y_test_est_ANN)
    yhat_lr.append(y_test_est_lr)
    yhat_baseline.append(np.ones(y_test.shape)*y_test.mean())
    y_true.append(y_test)

    Optimal_h_history[k] = opt_h
    Optimal_lambda_history[k] = opt_lambda
    k += 1

y_true = np.concatenate(y_true)
yhat_ANN = np.concatenate(yhat_ANN)
yhat_baseline = np.concatenate(yhat_baseline)
yhat_lr = np.concatenate(yhat_lr)

print("{:<40} {:<25} {:<25}".format("ANN", "LR", "Baseline"))
print("{:<28}  {:<25}".format("h_i", "lambda_i"))
for i in range(K_outter):
    print("{:<8} {:<20} {:<8} {:<15} {:<15}".format(Optimal_h_history[i][0], Error_test_ANN[i][0], Optimal_lambda_history[i][0], Error_test_rlr[i][0], Error_test_baseline[i][0]))


# SETUP I: ANN vs baseline, ANN vs lr, baseline vs lr
alpha = 0.01

z_ANN = np.abs(y_true - yhat_ANN ) ** 2
z_baseline = np.abs(y_true - yhat_baseline ) ** 2
z_lr = np.abs(y_true - yhat_lr) ** 2


z = z_ANN - z_baseline
CI_setupI_ANN_baseline = st.t.interval(1-alpha, len(z), loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p_setupI_ANN_baseline = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print("The confidence interval and p-value for ANN vs baseline in setup I are:")
print("Confidence interval:", CI_setupI_ANN_baseline)
print("p-value:", p_setupI_ANN_baseline)

z = z_baseline - z_lr
CI_setupI_lr_baseline = st.t.interval(1-alpha, len(z), loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p_setupI_lr_baseline = 2*st.t.cdf( -np.abs( np.mean(z) )/ st.sem(z), df=len(z)-1)  # p-value
print("\nThe confidence interval and p-value for lr vs baseline in setup I are:")
print("Confidence interval:", CI_setupI_lr_baseline)
print("p-value:", p_setupI_lr_baseline)

z = z_ANN - z_lr
CI_setupI_ANN_lr = st.t.interval(1-alpha, len(z), loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p_setupI_ANN_lr = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print("\nThe confidence interval and p-value for ANN vs lr in setup I are:")
print("Confidence interval:", CI_setupI_ANN_lr)
print("p-value:", p_setupI_ANN_lr)


# SETUP II
def correlated_ttest(r, rho, alpha=0.05):
    rhat = np.mean(r)
    shat = np.std(r)
    J = len(r)
    sigmatilde = shat * np.sqrt(1 / J + rho / (1 - rho))

    CI = st.t.interval(1 - alpha, df=J - 1, loc=rhat, scale=sigmatilde)  # Confidence interval
    p = 2*st.t.cdf(-np.abs(rhat) / sigmatilde, df=J - 1)  # p-value
    return p, CI

loss = 2

most_common_hidden_units = int(st.mode(Optimal_h_history, keepdims=True).mode[0][0])
most_common_lambda = st.mode(Optimal_lambda_history,  keepdims=True).mode[0][0]

K = 5
m = 1
J = 0

r_ANN_baseline = []
r_lr_baseline = []
r_ANN_lr = []

CV = model_selection.KFold(n_splits=K,shuffle=True, random_state = 43)

for dm in range(m):
    y_true = []
    yhat = []

    for train_index, test_index in CV.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        X_train_tensor = torch.tensor(X_train)
        y_train_tensor = torch.tensor(y_train)
        X_test_tensor = torch.tensor(X_test)
        y_test_tensor = torch.tensor(y_test)

        model_baseline = np.mean(y_test)
        
        model_lr = lm.Ridge(alpha=most_common_lambda).fit(X_train,y_train.squeeze()) # Linear least squares with l2 regularization.

        model_ANN = Net(most_common_hidden_units)
        model_ANN = model_ANN.to(device)

        net, final_loss, learning_curve = train_neural_net(model_ANN,
                                                       loss_fn,
                                                       X=X_train_tensor.to(device),
                                                       y=y_train_tensor.to(device),
                                                       n_replicates=3,
                                                       max_iter=max_iter)

        
        # Determine estimated regression value for test set
        yhat_baseline  = np.ones((y_test.shape[0],1))*model_baseline.squeeze()  
        yhat_lr =  model_lr.predict(X_test).reshape(-1,1)
        yhat_ANN = net(X_test_tensor.to(device)).cpu().detach().numpy()

        yhat.append( np.concatenate([yhat_baseline, yhat_ANN], axis=1) )
        y_true.append(y_test)        

        r_ANN_baseline.append(np.mean( np.abs( yhat_baseline-y_test ) ** loss - np.abs( yhat_ANN-y_test) ** loss ))
        r_lr_baseline.append(np.mean( np.abs( yhat_baseline-y_test ) ** loss - np.abs( yhat_lr-y_test) ** loss ))
        r_ANN_lr.append(np.mean( np.abs( yhat_ANN-y_test ) ** loss - np.abs( yhat_lr-y_test) ** loss ))

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K
p_setupII_ANN_baseline, CI_setupII_ANN_baseline = correlated_ttest(r_ANN_baseline, rho, alpha=alpha)
p_setupII_lr_baseline, CI_setupII_lr_baseline = correlated_ttest(r_lr_baseline, rho, alpha=alpha)
p_setupII_ANN_lr, CI_setupII_ANN_lr = correlated_ttest(r_ANN_lr, rho, alpha=alpha)


print(p_setupII_ANN_baseline, CI_setupII_ANN_baseline)
print(p_setupII_lr_baseline, CI_setupII_lr_baseline)
print(p_setupII_ANN_lr, CI_setupII_ANN_lr)