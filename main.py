import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


def removeOutliers(X, y):
    scaler = StandardScaler()
    X_new = scaler.fit_transform(X)

    clf = LocalOutlierFactor(n_neighbors=10)
    temp = clf.fit_predict(X_new)
    outliers = np.where(temp == -1)[0]
    
    X_clean = np.delete(X, outliers, axis=0)
    X_clean_df = pd.DataFrame(X_clean, columns=X.columns)
    y_clean = np.delete(y, outliers, axis=0)
    y_clean_df = pd.DataFrame(y_clean, columns=y.columns)

    return X_clean_df, y_clean_df


def kMeansClustering(X, data):
    kmeans = KMeans(n_clusters=20, random_state=0).fit(X)
    data['cluster'] = kmeans.labels_


def computePCA(X, num):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    pca = PCA(n_components=num)
    PCAx = pd.DataFrame(pca.fit_transform(X_train))
    return PCAx


def computeLDA(X_train, y_train, toBeComputed):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    xLDA = lda.transform(toBeComputed)

    return xLDA


def applyLogisticRegression(X_train, y_train, X_test, num):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    return y_pred


def applyRandomForestClassifier(X_train, y_train, X_test, num):
    y_train = (y_train.to_numpy()).ravel()
    rfc = RandomForestClassifier(n_estimators=num)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    return rfc, y_pred


def kFoldCrossValidation(model, X, y, k):
    kf = KFold(n_splits=k, shuffle=True)
    scores = cross_val_score(model, X, y, cv=kf)
    print("Accuracy: %0.2f" % (scores.mean()))


def saveToCSV(y, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'category'])
        for i, value in enumerate(y):
            writer.writerow([i, value])


dataTrain = pd.read_csv("D:\\IIIT Delhi\\4th Semester\\Courses\\Statistical Machine Learning\\Project\\data\\train.csv")
dataTrain = dataTrain.dropna()
dataTrain = dataTrain.drop(['ID'], axis=1)
X_train = dataTrain.drop(['category'], axis=1)
y_train = pd.DataFrame(dataTrain.iloc[:,-1].values)

dataTest = pd.read_csv("D:\\IIIT Delhi\\4th Semester\\Courses\\Statistical Machine Learning\\Project\\data\\test1.csv")
dataTest = dataTest.dropna()
X_test = dataTest.drop(['ID'], axis=1)

# kMeansClustering(X_train, dataTrain)

# X_clean, y_clean = removeOutliers(X_train, y_train)

# t = [X_clean, X_test]
# data = pd.concat(t)

# xPCA = computePCA(data, 600)

# X_train = xPCA.iloc[:1130, :]
# X_test = xPCA.iloc[1130:, :]


# lda = LinearDiscriminantAnalysis()
# lda.fit(X_train, y_clean)

# xLDA = lda.transform(X_train)
# x_testLDA = lda.transform(X_test)

y_pred = applyLogisticRegression(X_train, y_train, X_test, 10000)
# model, y_pred = applyRandomForestClassifier(xLDA, y_clean, x_testLDA, 100)

# kFoldCrossValidation(model, xLDA, y_clean, 5)
# print(y_pred)
saveToCSV(y_pred, "D:\\IIIT Delhi\\4th Semester\Courses\\Statistical Machine Learning\\Project\\test1.csv")
