from sklearn import tree
import pandas as pd
import numpy as np
from sklearn import preprocessing,cross_validation
from sklearn.utils import shuffle
from itertools import islice
from csv import reader
from csv import writer

# Load a CSV file
def load_csv(filename):
	# Do the reading
        file1 = open('output.csv', 'r')
        lines = reader(file1)
        new_rows_list = []
        for row in islice(lines,1,164302):
            new_row=list(row)
            if int(row[16]) != 1:
                new_row[16]= 0
            new_rows_list.append(new_row)
        file1.close()   # <---IMPORTANT

        # Do the writing
        file2 = open(filename, 'w' ,newline='')
        lines = writer(file2)
        lines.writerows(new_rows_list)
        file2.close()
        
#Reading training data set
filename = 'accident_dt_noinjury.csv'
load_csv(filename)
df = pd.read_csv(filename)

#Dividing it into X and y
X = np.array(df.drop(df.columns[[1,16,17]],1)) 
y = np.array(df.iloc[:,16:17])
print(y)

#Preprocessing scales the feature values. Cross validation shuffles them and splits into train(80%) and test(20%) datasets
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = tree.DecisionTreeClassifier(criterion="gini",max_depth=10)

clf = clf.fit(X_train, y_train)
tree.export_graphviz(clf,out_file='tree.dot')


accuracy = clf.score(X_test,y_test)*100
error=100-accuracy
print("Accuracy = %f" %accuracy)
print("Train Error = %f" %error)

