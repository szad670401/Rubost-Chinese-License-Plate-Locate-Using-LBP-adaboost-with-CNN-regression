import cv2
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import theano

import numpy as np





def norm(img,datashape=(13,55)):
    img = cv2.resize(img, (datashape[1], datashape[0]));
    #flag,img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 5, 0.5)
    #
    # cv2.imshow("imgx",img);
    # cv2.waitKey(0)
    img = (img.astype(np.float32) / 255)
    shape = img.shape
    if len(shape) == 2:
        img -= img.mean()
        img = np.expand_dims(img, 2)
    else:
        img[:,:,1] -= img[:,:,1].mean()




    return img



def loadData(pathT,pathF,datashape=(13,55)):
    prepare_data =[]
    prepare_label = [];
    for parent,dirnames,filenames in os.walk(pathT):
        for filename in filenames:
            path = os.path.join(parent,filename)
            if path.endswith(".jpg") or path.endswith(".png") or path.endswith(".bmp"):
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = norm(img,datashape)
                prepare_data.append(img);
                prepare_label.append([1,0]);

    for parent, dirnames, filenames in os.walk(pathF):
        for filename in filenames:
            path = os.path.join(parent, filename)
            if path.endswith(".jpg") or path.endswith(".png") or path.endswith(".bmp"):
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = norm(img,datashape)

                prepare_data.append(img);
                prepare_label.append([0,1]);

    return np.array(prepare_data),np.array(prepare_label)




def arrangeData(data):
    prepare_data, prepare_label = data;
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(prepare_data))
    shuffle = np.random.permutation(len(prepare_data));

    digits, labels = prepare_data[shuffle], prepare_label[shuffle]
    return digits,labels



def constructmodel(inputshape):
    model = Sequential()
    extract_conv1 = Convolution2D(8, 3, 3, border_mode='valid', input_shape=(inputshape))
    model.add(extract_conv1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(16, 2, 2, border_mode='valid'))
    # model.add(Convolution2D(64, 1, 1, border_mode='valid'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))

    # model.add(Dense(32, init='normal'))
    # model.add(Activation('relu'))
    model.add(Dense(2, init='normal'))
    model.add(Activation('softmax'))

    return model

def train(pathT,pathF):
    model = constructmodel((13,55,1))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=["accuracy"])

    data = loadData(pathT,pathF);

    training_data,training_label  = arrangeData(data)
    print training_label.shape,training_data.shape
    print training_label
    model.fit(training_data,training_label,nb_epoch=50,validation_split=0.1,show_accuracy=True)
    model.save("./judge1.h5")








#
#
#train("./training_T","./training_F")
#
