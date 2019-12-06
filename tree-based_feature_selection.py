from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt

df=ps.read_csv('dataset.csv')

X = df.drop(['CASENUM','MAXSEV_I'],1)
y = df.iloc[:,31:32]

y=np.ravel(y,order="C")
xlabels=X.columns
print(xlabels)
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
print(clf.feature_importances_)  
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)
#print(X_new[:,8:12])
#plot
N = len(clf.feature_importances_)
x = range(N)
width=2/3
ax=plt.bar(x,clf.feature_importances_,width,color='blue')
plt.title('Tree Based Feature Selection')
plt.xticks(x,xlabels, rotation='vertical')
plt.legend()
plt.show()

