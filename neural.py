import cv2
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import csv
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier



def column(matrix, i):
    return [row[i] for row in matrix]

X1=[]
i=0
f=open('train.csv','rt')
try:
    reader=csv.reader(f)

    for row in reader:
        if i>0:
            row=map(int,row)
            #row=row/255
            X1.append(row)
        i=i+1
        #if i>2000:
        #    break

finally:
    f.close()


X1=np.asarray(X1)

y=column(X1,0)
print X1,y
print len(y)
X1 = np.delete(X1, 0, axis=1)

max_abs_scaler = preprocessing.MaxAbsScaler()
X1 = max_abs_scaler.fit_transform(X1)
print X1


print X1.shape

# for row in X1:
#     print row

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10],'gamma':[0,10]}

#clf=svm.SVC()
#clf = GridSearchCV(clf, parameters)
#print clf.fit(X1,y)

# clf=SGDClassifier(loss="hinge",penalty="l2",n_iter=100)
# clf.fit(X1,y)

clf = MLPClassifier(hidden_layer_sizes=(784,10), max_iter=1000, alpha=0.001, verbose=10, tol=1e-8, random_state=1,
                    learning_rate_init=.01)
clf.fit(X1,y)



X2=[]
i=0
f=open('test.csv','rt')
try:
    reader=csv.reader(f)
    for row in reader:
        if i>0:
            row=map(int,row)
            X2.append(row)
        i=i+1
        print i

finally:
    f.close()



X2=np.asarray(X2)
blah=[]

fr=open('output3.csv','w')
fr.write("ImageId"+',Label')
fr.write('\n')
for i in range(28000):
    fr.write(str(i+1))
    fr.write(',')
    X2[i] = X2[i].reshape(1, -1)
    #print X2[i]
    ans=clf.predict(X2[i])
    fr.write(str(ans[0]))
    fr.write('\n')
    print i


print ans
