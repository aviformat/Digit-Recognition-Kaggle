import cv2
import numpy as np
from sklearn import svm
import csv
import pandas as pd
from sklearn.model_selection import GridSearchCV


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
            X1.append(row)
        i=i+1

finally:
    f.close()


X1=np.asarray(X1)

y=column(X1,0)
print X1,y
print len(y)
X1 = np.delete(X1, 0, axis=1)
print X1.shape

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10],'gamma':[0,10]}

clf=svm.SVC()
clf = GridSearchCV(clf, parameters)
print clf.fit(X1,y)

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

fr=open('output1.csv','w')
fr.write("ImageId"+',Label')
fr.write('\n')
for i in range(28000):
    fr.write(str(i+1))
    fr.write(',')
    X2[i] = X2[i].reshape(1, -1)
    print X2[i]
    ans=clf.predict(X2[i])
    fr.write(str(ans[0]))
    fr.write('\n')
    print i


print ans







