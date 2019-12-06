import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as ps


df =ps.read_csv("accident.csv",nrows=50000)

X = df.drop(['CASENUM','MAX_SEV','MAXSEV_I'],1)
y = df.iloc[:,48:49]

xlabels=X.columns
print(X.shape)
y=np.ravel(y,order="C")
X.dropna(inplace="True")
sel = SelectKBest(chi2,k=5)
X_new = sel.fit_transform(X,y)
print(X_new)
print(sel.pvalues_)
scores = -np.log10(sel.pvalues_)
#plt.bar(range(30),scores)
#plt.xticks(range(30), X, rotation='vertical')
#plt.show()


N = len(sel.scores_)
x = range(N)
width=1/1.5
plt.title('SelectKBest Feature Selection')
plt.bar(x,sel.scores_,width,color='blue')
plt.xticks(x,xlabels, rotation='vertical')
plt.show()

