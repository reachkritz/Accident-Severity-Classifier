import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
        
#Reading training data set
filename = 'dataset - part1.csv'
df = pd.read_csv(filename)

X = df.drop(['YEAR','MAXSEV_I'],1)
y = df['MAXSEV_I']
#print(X)
#print(y)

xlabels=X.columns
print(X.shape)
y=np.ravel(y,order="C")
X.dropna(inplace="True")
sel = SelectKBest(chi2,k=5)
X_new = sel.fit_transform(X,y)
print(X_new)
#print(X.iloc[:,0])

scores = -np.log10(sel.pvalues_)


N = len(sel.scores_)
x = range(N)
width=1/1.5
plt.bar(x,sel.scores_,width,color='blue')
plt.xticks(x,xlabels, rotation='vertical')
plt.show()

#adding new column
filename2 = 'dataset - combofeatures.csv'
df = pd.DataFrame() 
new_column = X.iloc[:,0] * sel.pvalues_[0]
for x in range(1,10):
        print(sel.pvalues_[x])
        new_column += X.iloc[:,x] * sel.pvalues_[x] 
df['NewColumn1'] = new_column

df.to_csv(filename2, sep=',')
#new_column = df[i]*sel.pva
# we then add the series to the dataframe, which holds our parsed CSV file
#df['NewColumn'] = new_column
# save the dataframe to CSV
#df.to_csv('path/to/file.csv', sep='\t')



#PART 2
filename = 'dataset - part2.csv'
df = pd.read_csv(filename)

X = df.drop(['SPDLIM_H','MAXSEV_I'],1)
y = df['MAXSEV_I']
#print(X)
#print(y)

xlabels=X.columns
print(X.shape)
y=np.ravel(y,order="C")
X.dropna(inplace="True")
sel = SelectKBest(chi2,k=5)
X_new = sel.fit_transform(X,y)
print(X_new)
print(sel.pvalues_)
scores = -np.log10(sel.pvalues_)


N = len(sel.scores_)
x = range(N)
width=1/1.5
plt.bar(x,sel.scores_,width,color='blue')
plt.xticks(x,xlabels, rotation='vertical')
plt.show()

#adding new column
filename2 = 'dataset - combofeatures.csv'
df = pd.read_csv(filename2)
new_column = X.iloc[:,0] * sel.pvalues_[0]
for x in range(1,10):
        print(sel.pvalues_[x])
        new_column += X.iloc[:,x] * sel.pvalues_[x] 
df['NewColumn2'] = new_column

df.to_csv(filename2, sep=',')

#PART 3
filename = 'dataset - part3.csv'
df = pd.read_csv(filename)

X = df.drop(['MAXSEV_I'],1)
y = df['MAXSEV_I']
#print(X)
#print(y)

xlabels=X.columns
print(X.shape)
y=np.ravel(y,order="C")
X.dropna(inplace="True")
sel = SelectKBest(chi2,k=5)
X_new = sel.fit_transform(X,y)
print(X_new)
print(sel.pvalues_)
scores = -np.log10(sel.pvalues_)


N = len(sel.scores_)
x = range(N)
width=1/1.5
plt.bar(x,sel.scores_,width,color='blue')
plt.xticks(x,xlabels, rotation='vertical')
plt.show()

#adding new column
filename2 = 'dataset - combofeatures.csv'
df = pd.read_csv(filename2)
new_column = X.iloc[:,0] * sel.pvalues_[0]
for x in range(1,10):
        print(sel.pvalues_[x])
        new_column += X.iloc[:,x] * sel.pvalues_[x] 
df['NewColumn3'] = new_column

df.to_csv(filename2, sep=',')
