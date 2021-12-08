from scipy.io import arff
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def loaddata(filename):    
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    # df = df.dropna()
    labels = df['class'].values.astype(np.int)

    # labels = [int(item) for item in labels]
    
    # print(labelList)
    df = df.drop(labels='class', axis=1) # axis 1 drops columns, 0 will drop rows that match index value in labels
    # print(df.values)
    embeddings = df.values
    return labels,embeddings

def datapreprocess(embeddings, labels):
    # set all nan value to 0, useless
    embeddings[np.isnan(embeddings)]=0
    # embeddings = np.delete(embeddings,36,1)
    # embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]
    
    
    # # do some PCA analysis to reduce the dimention of the data
    # # n_components = 16/32/48
    # pca = PCA(n_components=48)
    # embeddings = pca.fit_transform(embeddings)
    
    # select the best arributes of the data, for some arributes are highly correlated
    # this method almost has the same effect with PCA, but it does not re-compute the data, 
    # just delete some useless attributes to reduce the dimention
    # before use this method, use sigmoid, for the SelectKBest cannot fit negative values
    embeddings = sigmoid(embeddings)
    embeddings = SelectKBest(chi2, k=60).fit_transform(embeddings, np.array(labels))
    
    return embeddings

def generatePCAMap(labels,embeddings,embeddingFileName):
    colorList = []
    for item in labels:
        if(int(item) == 0):
            colorList.append('m')
        else:
            colorList.append('k')
            
    pca = PCA(n_components=5)
    reduced = pca.fit_transform(embeddings)

    t = reduced.transpose()
    plt.scatter(t[0], t[1],s=1,c=colorList,linewidths=0)

    plt.title(embeddingFileName)

    plt.savefig(embeddingFileName + '.jpg', format='jpg', dpi=1000)
    print('saved '+ embeddingFileName + '.jpg')

def classificationProcess(clf, X_train, y_train, X_test, y_test, modelName):
    # start = time.process_time()
    start = time.perf_counter()
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    end = time.perf_counter()
    # print("%s: Number of mislabeled points out of a total %d points : %d"
    #    % (modelName,X_test.shape[0], (y_test != y_pred).sum()))
    acc = (y_test == y_pred).sum()/X_test.shape[0]
    recall = recall_score(y_test, y_pred)
    ROC_AUC = roc_auc_score(y_test, y_pred)
    PR_AUC = average_precision_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred, average='macro')
    time_used = end-start
    print("{} Accuracy is {}".format(modelName, acc))
    print("{} Recall is {}, {}/{}".format(modelName, recall, ((y_pred == 1)&(y_test == 1)).sum(),(y_test == 1).sum()))
    print("{} ROC AUC is {}".format(modelName, ROC_AUC))
    print("{} PR AUC is {}".format(modelName, PR_AUC))    
    print("{} F1-score is {}".format(modelName, F1_score))
    print("{} time used is {}".format(modelName, time_used))
    with open("log.txt",'a') as f:
        f.writelines("| {} | {:.3f} |{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |\n".format(modelName,acc,recall,ROC_AUC,PR_AUC,F1_score,time_used))
    
def main():
    # labels = []
    # embeddings = []
    for i in range(1,6):
        with open("log.txt",'a') as f:
            f.writelines("\n\n### Table {}year.arff\n".format(i))
            f.writelines("| ModelName | Precision | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |\n")
            f.writelines("| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |\n")
        embeddingFileName = "./data/{}year.arff".format(i)
        labels,embeddings = loaddata(embeddingFileName)
        # labels.extend(label.tolist())
        # embeddings.extend(embedding.tolist())
        # embeddings = np.vstack((embedding,embeddings))
        
        embeddings = np.array(embeddings)
        print(embeddings.shape)
        embeddings = datapreprocess(embeddings, labels)
        print(embeddings.shape)
        generatePCAMap(labels,embeddings,"./data/total")
            
        
        X = embeddings
        y = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        
        # Bernoulli Naive Bayes
        clf = BernoulliNB()
        classificationProcess(clf, X_train, y_train, X_test, y_test, "BernoulliNB")
        
        # Gaussian Naive Bayes
        clf = GaussianNB()
        classificationProcess(clf, X_train, y_train, X_test, y_test, "GaussianNB")
        
        # SVM
        clf = svm.SVC()
        classificationProcess(clf, X_train, y_train, X_test, y_test, "SVM")
        
        
        # DecisionTree
        clf = tree.DecisionTreeClassifier()
        classificationProcess(clf, X_train, y_train, X_test, y_test, "DecisionTree")
        
        # Stochastic Gradient Descent
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=50)
        classificationProcess(clf, X_train, y_train, X_test, y_test, "SGD")
        
        # Nearest Neighbors
        clf = NearestCentroid()
        classificationProcess(clf, X_train, y_train, X_test, y_test, "NearestCentroid")
        
        # AdaBoost
        clf = AdaBoostClassifier(n_estimators=100)
        classificationProcess(clf, X_train, y_train, X_test, y_test, "AdaBoost")
        
        # GradientBoosting
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classificationProcess(clf, X_train, y_train, X_test, y_test, "GradientBoosting")
        
        # HistGradientBoosting
        clf = HistGradientBoostingClassifier(max_iter=100)
        classificationProcess(clf, X_train, y_train, X_test, y_test, "HistGradientBoosting")
        
        # Neural network models (supervised)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, ), random_state=1, max_iter=1000)
        classificationProcess(clf, X_train, y_train, X_test, y_test, "MLP")
    
      
    
if __name__ == "__main__":
    main()
 