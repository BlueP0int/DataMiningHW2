import numpy as np
import pandas as pd
import math
import datetime
import time
import random

from matplotlib import pyplot
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score

from imblearn.over_sampling import SMOTE


def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average='macro')

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loaddata(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    labels = df['class'].values.astype(int)

    df = df.drop(labels='class', axis=1) # axis 1 drops columns, 0 will drop rows that match index value in labels
    embeddings = df.values
    return labels, embeddings


def datapreprocess(embeddings, labels):
    embeddings[np.isnan(embeddings)] = 0
    embeddings = sigmoid(embeddings)
    embeddings = SelectKBest(chi2, k=48).fit_transform(embeddings, np.array(labels))

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

def border_line_1_smote(X, y, k, minor_class=1, sample_ratio=1):
    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(X)
    new_samples = []
    for index, label in enumerate(y):
        if label == minor_class:
            knn = neigh.kneighbors([X[index]], return_distance=False)[0]
            count = sum([1 if y[j] != minor_class else 0 for j in knn[1:]])
            if count >= k / 2 and count != k:
                minor = []
                for j in knn[1:]:
                    if y[j] == minor_class:
                        minor.append(X[j])
                x = X[index]
                for dummy in range(sample_ratio):
                    x1 = random.sample(minor, 1)[0]
                    new_samples.append(x + (x1 - x) * random.random())
    return np.concatenate([X, new_samples], axis=0), np.concatenate([y, [minor_class for i in range(len(new_samples))]], axis=0)

def border_line_2_smote(X, y, k, minor_class=1, sample_ratio=1):
    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(X)
    new_samples = []
    for index, label in enumerate(y):
        if label == minor_class:
            knn = neigh.kneighbors([X[index]], return_distance=False)[0]
            count = sum([1 if y[j] != minor_class else 0 for j in knn[1:]])
            if count >= k / 2 and count != k:
                minor = [X[j] for j in knn[1:]]
                x = X[index]
                for dummy in range(sample_ratio):
                    x1 = random.sample(minor, 1)[0]
                    new_samples.append(x + (x1 - x) * random.random())
    return np.concatenate([X, new_samples], axis=0), np.concatenate([y, [minor_class for i in range(len(new_samples))]], axis=0)

def tomek_link(X, y):
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(X)
    removed = {}
    nn = []
    for i in range(len(X)):
        nn.append(neigh.kneighbors([X[i]], return_distance=False)[0][1])
    for i, j in enumerate(nn):
        if y[i] != y[j] and nn[j] == i:
            removed[i] = ""
            removed[j] = ""
    ret_X = []
    ret_y = []
    for i in range(len(X)):
        if i not in removed:
            ret_X.append(X[i])
            ret_y.append(y[i])
    return np.array(ret_X), np.array(ret_y)

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


def experiment(i, cost_gradient_func, threshold=0.5, alpha=0.05, max_iter=400, stop_threshold=0.0001, mode=None, ratio=1):
    y, X = loaddata("./data/{}year.arff".format(i))
    y = np.array(y)
    X = np.array(X)
    X = datapreprocess(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_test = concat_zero_column(X_test)

    if mode == "smote":
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    elif mode == "smote1":
        X_train, y_train = border_line_1_smote(X_train, y_train, 3, sample_ratio=ratio)
    elif mode == "smote2":
        X_train, y_train = border_line_2_smote(X_train, y_train, 3, sample_ratio=ratio)
    elif mode == "pipeline1":
        X_train, y_train = border_line_1_smote(X_train, y_train, 3, sample_ratio=ratio)
        X_train, y_train = tomek_link(X_train, y_train)
    elif mode == "pipeline2":
        X_train, y_train = border_line_2_smote(X_train, y_train, 3, sample_ratio=ratio)
        X_train, y_train = tomek_link(X_train, y_train)

    X_train = concat_zero_column(X_train)

    start = time.perf_counter()
    theta = gradient_descent(X_train, y_train, np.zeros(X_train.shape[1]), cost_gradient_func, alpha=alpha, max_iter=max_iter, stop_threshold=stop_threshold)
    if threshold is None:
        threshold = sum(y_train) / len(y_train) #+ 0.05
    y_pred = sigmoid(X_test.dot(theta)) > threshold
    end = time.perf_counter()
    ACC = (y_pred == y_test).sum() / y_test.size
    recall = recall_score(y_test, y_pred)
    ROC_AUC = roc_auc_score(y_test, y_pred)
    PR_AUC = average_precision_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred, average='macro')
    F2_score = f2_score(y_test, y_pred)
    time_used = end-start

    print("ACC: {}, Recall: {}, ROC_AUC: {}, PR_AUC: {}, F1: {}, Time: {}".format(ACC, recall, ROC_AUC, PR_AUC, F1_score, time_used))
    print("recall is {}, {}/{}".format(recall, ((y_pred == 1)&(y_test == 1)).sum(),(y_test == 1).sum()))
    with open("log.txt",'a') as f:
        f.writelines("| {} | {:.3f} |{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |\n".format("{}".format(str(mode)),ACC,recall,ROC_AUC,PR_AUC,F1_score, F2_score, time_used))

def main():
    #thresholds = [0.022, 0.033, 0.04, 0.05, 0.06]
    thresholds = [0.5, 0.5, 0.5, 0.5, 0.5]
    ratios = [60, 50, 30, 30, 20]
    for i in range(1, 6):
        with open("log.txt", 'a') as f:
            f.writelines("| method | Accuracy |Recall | ROC_AUC | PR_AUC | F1 | F2 | Time |\n")
            f.writelines("| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  | :---: |\n")
        experiment(i, cost_gradient, alpha=0.025, max_iter=600)
        experiment(i, cost_gradient, alpha=0.025, max_iter=600, mode="smote")
        experiment(i, cost_gradient, alpha=0.025, max_iter=600, mode="smote1", ratio=ratios[i - 1])
        experiment(i, cost_gradient, alpha=0.025, max_iter=600, mode="smote2", ratio=ratios[i - 1])
        experiment(i, cost_gradient, alpha=0.025, max_iter=600, mode="pipeline1", ratio=ratios[i - 1])
        experiment(i, cost_gradient, alpha=0.025, max_iter=600, mode="pipeline2", ratio=ratios[i - 1])


if __name__ == "__main__":
    main()
