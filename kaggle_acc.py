from keras.models import Sequential,Input
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.optimizers import SGD,RMSprop
import pandas as pd
from keras.losses import mean_squared_error,categorical_crossentropy,mean_squared_error
from keras.utils import plot_model
# from keras.models import
from keras.models import load_model
import pandas as pd
import os
import cv2
import csv
import numpy as np

X_test=pd.read_csv("test.csv")

X,y=[],[]
print X_test.shape
for i in range(28000):
    X.append(np.reshape(np.asarray(X_test.iloc[i]),(-1,28)))
    # y.append(y_train.iloc[i])
X=np.asarray(X)
nr,nx,ny=X.shape
X=np.reshape(X,(nr,nx,ny,1))
print "done"
model = load_model('digits_kaggle.h5')
ans=model.predict(X)
print ans

f=open("kaggle_ans.csv",'wt')
writer=csv.writer(f)
writer.writerow(('ImageId','Label'))
for i in range(28000):
    max_index = max(range(len(ans[i])), key=lambda index: ans[i][index])
    writer.writerow((str(i+1),str(max_index)))