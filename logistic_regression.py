import numpy as np
import pandas as pd
import math
import datetime
import time

from matplotlib import pyplot
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

from imblearn.over_sampling import SMOTE


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loaddata(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    df = df.dropna()
    labels = df['class'].values.astype(int)

    df = df.drop(labels='class', axis=1)  # axis 1 drops columns, 0 will drop rows that match index value in labels
    embeddings = df.values
    return labels, embeddings


def datapreprocess(embeddings, labels):
    # set all nan value to 0, useless
    # embeddings[np.isnan(embeddings)]=0
    # embeddings = np.delete(embeddings,36,1)
    #embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]
    embeddings[np.isnan(embeddings)] = 0
    embeddings = sigmoid(embeddings)

    ## do some PCA analysis to reduce the dimention of the data
    ## n_components = 16/32/48
    # pca = PCA(n_components=16)
    # embeddings = pca.fit_transform(embeddings)

    # select the best arributes of the data, for some arributes are highly correlated
    # this method almost has the same effect with PCA, but it does not re-compute the data,
    # just delete some useless attributes to reduce the dimention
    # before use this method, use sigmoid, for the SelectKBest cannot fit negative values
    embeddings = SelectKBest(chi2, k=60).fit_transform(embeddings, np.array(labels))

    return embeddings


def cost_gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    h_x = sigmoid(X.dot(theta))
    J = 0
    for idx, h_x_i in enumerate(h_x):
        J += -y[idx] * math.log(h_x_i) - (1 - y[idx]) * math.log(1 - h_x_i)
    J /= y.size
    for j, diff in enumerate(np.subtract(h_x, y)):
        grad += diff * X[j]
    grad /= y.size
    return J, grad


def cost_gradient_reg(X, y, theta, lambda_=1):
    J, grad = cost_gradient(X, y, theta)
    for i in range(1, theta.shape[0]):
        J += (theta[i] ** 2) * lambda_ / (2 * y.size)
        grad[i] += theta[i] * lambda_ / y.size
    return J, grad


def concat_zero_column(X):
    m, n = X.shape
    return np.concatenate([np.ones((m, 1)), X], axis=1)


def data_split(X, y, val=0.1, test=0.2):
    X_other, X_val, y_other, y_val = train_test_split(X, y, test_size=val, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-val)*test, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test


def gradient_descent(X, y, initial_theta, cost_grad_func, alpha=0.025, max_iter=100, stop_threshold=0.0001):
    theta = initial_theta
    last_cost = 0
    for i in range(max_iter):
        cost, grad = cost_grad_func(X, y, theta)
        theta = theta - alpha * grad
        if abs(last_cost - cost) < stop_threshold:
            break
        last_cost = cost
    return theta


def plot_loss(X_train, y_train, X_val, y_val, cost_grad_func, alpha=0.025, max_iter=200, title=None):
    theta = np.zeros(X_train.shape[1])
    cost_train = []
    cost_val = []
    for i in range(max_iter):
        cost, grad = cost_grad_func(X_train, y_train, theta)
        cost1, foo = cost_grad_func(X_val, y_val, theta)
        cost_train.append(cost)
        cost_val.append(cost1)
        theta = theta - alpha * grad
    ps = []
    p, = pyplot.plot(np.arange(max_iter), cost_train, lw=2)
    ps.append(p)
    p, = pyplot.plot(np.arange(max_iter), cost_val, lw=2)
    ps.append(p)
    labels = ["train", "val"]
    pyplot.legend(handles=ps, labels=labels)
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Loss')
    if title is not None:
        pyplot.title(title)
        pyplot.savefig(title + '.jpg', format='jpg', dpi=1000)

def predict(X, theta, threshold=0.5):
    return sigmoid(X.dot(theta)) > threshold

def choose_model(i, cost_gradient_func, thresholds, alpha=0.05, max_iter=400, stop_threshold=0.0001):
    y, X = loaddata("./data/{}year.arff".format(i))
    y = np.array(y)
    X = np.array(X)
    X = datapreprocess(X, y)
    X = concat_zero_column(X)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(X, y)
    
    accs = []
    recalls = []
    
    theta = gradient_descent(X_train, y_train, np.zeros(X_train.shape[1]), cost_gradient_func, alpha=alpha, max_iter=max_iter, stop_threshold=stop_threshold)
    for threshold in thresholds:
        y_pred = predict(X_test, theta, threshold)
        accs.append((y_pred == y_test).sum() / y_test.size)
        recalls.append(recall_score(y_test, y_pred))
    
    return theta, accs, recalls


def experiment(i, cost_gradient_func, threshold=0.5, alpha=0.05, max_iter=400, stop_threshold=0.0001, title=None):
    y, X = loaddata("./data/{}year.arff".format(i))
    y = np.array(y)
    X = np.array(X)
    X = datapreprocess(X, y)
    X = concat_zero_column(X)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(X, y)

    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    #fig = pyplot.figure()
    #plot_loss(X_train, y_train, X_val, y_val, cost_gradient_func, title=title)

    start = time.perf_counter()
    theta = gradient_descent(X_train, y_train, np.zeros(X_train.shape[1]), cost_gradient_func, alpha=alpha, max_iter=max_iter, stop_threshold=stop_threshold)
    y_pred = sigmoid(X_test.dot(theta)) > threshold
    end = time.perf_counter()
    ACC = (y_pred == y_test).sum() / y_test.size
    recall = recall_score(y_test, y_pred)
    ROC_AUC = roc_auc_score(y_test, y_pred)
    PR_AUC = average_precision_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred, average='macro')
    time_used = end-start

    print("ACC: {}, Recall: {}, ROC_AUC: {}, PR_AUC: {}, F1: {}, Time: {}".format(ACC, recall, ROC_AUC, PR_AUC, F1_score, time_used))
    print("recall is {}, {}/{}".format(recall, ((y_pred == 1)&(y_test == 1)).sum(),(y_test == 1).sum()))
    with open("log.txt",'a') as f:
        f.writelines("| {} | {:.3f} |{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |\n".format("LogisticRegression(Ours)",ACC,recall,ROC_AUC,PR_AUC,F1_score,time_used))

def main():
    #thresholds = [0.022, 0.033, 0.04, 0.05, 0.06]
    thresholds = [0.5, 0.5, 0.5, 0.5, 0.5]
    for i in range(1, 6):
        experiment(i, cost_gradient, title="{}year".format(i), threshold=thresholds[i-1], alpha=0.025, max_iter=600)


if __name__ == "__main__":
    main()
