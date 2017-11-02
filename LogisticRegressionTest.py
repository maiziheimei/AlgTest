#Logistic Regression example from
# https://medium.com/towards-data-science/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import logistic

import matplotlib.pyplot as plt
plt.rc("font", size=14)

from sklearn import model_selection
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#training data info
data=pd.read_csv('resources/banking.csv',header=0)
data=data.dropna()

data.drop(data.columns[[0,3,7,8,9,10,11,12,13,15,16,17,18,19]],axis=1, inplace=True)
data.head()

data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
data2.drop(data2.columns[[12,16,18,21,24]], axis=1, inplace=True)
data2.head()
data2.columns

sns.heatmap(data2.corr())
plt.show()

#split the data into training and test sets
X = data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)

X_train.shape

X_test.shape

# LR model
classifier = logistic.LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#predicating the test set results and create confusion matrix
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix= confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#accuracy
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

#compute precistion, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# classifier visulization playground

from sklearn.decomposition import PCA
X = data2.iloc[:,1:]
y = data2.iloc[:,0]
pca = PCA(n_components=2).fit_transform(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(pca, y, random_state=0)
plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='YES', s=2, color='navy')
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='NO', s=2, color='darkorange')
plt.legend()
plt.title('Bank Marketing Data Set\nFirst Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().set_aspect('equal')
plt.show()

# #2 classifier visulization playground
# import sys
# def plot_bank(X, y, fitted_model):
#     plt.figure(figsize=(9.8,5), dpi=100)
#     for i, plot_type in enumerate(['Decision Boundary', 'Decision Probabilities']):
#         plt.subplot(1,2,i+1)
#
#     mesh_step_size = 0.01  # step size in the mesh
#     x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
#     y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))
#     if i == 0:
#         Z =  fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])
#     else:
#         try:
#             Z = fitted_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
#         except:
#             plt.text(0.4, 0.5, 'Probabilities Unavailable', horizontalalignment='center',verticalalignment='center', transform = plt.gca().transAxes, fontsize=12)
#             plt.axis('off')
#             sys.exit()#break
#     Z = Z.reshape(xx.shape)
#     plt.scatter(X[y.values==0,0], X[y.values==0,1], alpha=0.8, label='YES', s=5, color='navy')
#     plt.scatter(X[y.values==1,0], X[y.values==1,1], alpha=0.8, label='NO', s=5, color='darkorange')
#     plt.imshow(Z, interpolation='nearest', cmap='RdYlBu_r', alpha=0.15,extent=(x_min, x_max, y_min, y_max), origin='lower')
#     plt.title(plot_type + '\n' +str(fitted_model).split('(')[0]+ ' Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 5)))
#     plt.gca().set_aspect('equal');
#     plt.tight_layout()
#     plt.legend()
#     plt.subplots_adjust(top=0.9, bottom=0.08, wspace=0.02)
#
# model = logistic.LogisticRegression()
# model.fit(X_train,y_train)
#
# plot_bank(X_test, y_test, model)
# plt.show()