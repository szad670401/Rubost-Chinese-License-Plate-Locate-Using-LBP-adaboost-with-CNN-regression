#coding=utf-8
import numpy as np
import cv2
import os
import time
import training_judging as tj;
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import keras
import numpy as np

watch_cascade = cv2.CascadeClassifier('./out-long/cascade.xml')
count = 0;


model = tj.constructmodel((13,55,1))

model.load_weights("./judge1.h5")


extend_scale = 1;

shape = [140,180]


def getTransformMatrix(origin_shape,shape_= (140,180)):
    pts1 = np.float32([[0,0],[origin_shape[1],0],[origin_shape[1],origin_shape[0]],[0,origin_shape[0]]])
    pts2 = np.float32([[0,0],[shape_[1],0],[shape_[1],shape_[0]],[0,shape_[0]] ])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return M


def resize_with_points(pts,shape,image):
    M = getTransformMatrix(image.shape,shape)
    points_ = np.dot(M, np.float32(np.vstack((pts.T, np.array([1, 1, 1, 1]).T))))[0:2, 0:4].T.astype(np.int)
    return points_


def foutpoint(p_points):
    p_points = p_points[0]
    xp = np.array([[p_points[0]*shape[1],p_points[1]*shape[0]],
                  [p_points[2] * shape[1], p_points[3] * shape[0]],
                  [p_points[4] * shape[1], p_points[5] * shape[0]],
                  [p_points[6] * shape[1], p_points[7] * shape[0]]
                  ])
    return xp


def drawPoints(img,p_points):
    # img = img*255
    # img = img.astype(np.uint8);

    # p = p_points.reshape([4,2]);
    # r = r_points.reshape([4, 2]);
    # print p
    p_points = p_points[0]
    # print p_points
    p = np.array([[p_points[0]*shape[1],p_points[1]*shape[0]],
                  [p_points[2] * shape[1], p_points[3] * shape[0]],
                  [p_points[4] * shape[1], p_points[5] * shape[0]],
                  [p_points[6] * shape[1], p_points[7] * shape[0]]
                  ])
    # print p

    # r = np.array([[r_points[0]*200,r_points[1]*72],
    #               [r_points[2] * 200, r_points[3] * 72],
    #               [r_points[4] * 200, r_points[5] * 72],
    #               [r_points[6] * 200, r_points[7] * 72]
    #               ])
    # points[0][0]  = p_points[0] 200, points[0][1] / 72,
    # points[1][0] / 200, points[1][1] / 72,
    # points[2][0] / 200, points[2][1] / 72,
    # points[3][0] / 200, points[3][1] / 72]
    for one in p:
        cv2.circle(img,(int(one[0]),int(one[1])),1,(0,255,0),2)
    # print "R",r
    # for one in r:
    #     cv2.circle(img,(int(one[0]),int(one[1])),3,(0,255,0))
    cv2.imshow("img", img)
    cv2.waitKey(0)

def rectSrceen(pts,rect):
    zeroadd = np.array(rect[:2])
    return np.array(pts) + zeroadd


def norm(X):
    #X = cv2.resize(X,(28,28))
    P = cv2.cvtColor(X,cv2.COLOR_RGB2GRAY)
    P =  P.astype(np.float32)/255
    P = np.expand_dims(P,2)
    return P


def getmodel(path):
    model = Sequential();
    model.add(Convolution2D(1, 7, 7, border_mode='valid', input_shape=(140, 180, 1), subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Convolution2D(1, 3, 3, border_mode='valid', subsample=(2, 2)))
    model.add(Convolution2D(32, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(128, init='normal'))
    model.add(Activation('relu'))
    model.add(Dense(8, init='normal'))
    model.add(Activation('relu'))
    model.load_weights(path)
    return model

def packFilename(origin_filename,position,type=".png"):
    name,suffix = os.path.splitext(origin_filename)
    filename = name+"_"
    for coor in position:
        filename+=str(coor)+"_"
    filename+=type;
    return filename







models = getmodel("model_t5_regression.h5")

path = "./general_test"

for parent,dirnames,filenames in os.walk("/Users/yujinke/Desktop/EasyPR/resources/image/general_test"):

    for filename in filenames:
    # while(1):
    #     filename = filenames[np.random.randint(0,len(filenames))]
        path = os.path.join(parent,filename)

        if path.endswith(".jpg"):
            img = cv2.imread(path)
            img_bak  = cv2.imread(path,cv2.IMREAD_GRAYSCALE  )
            img_bak = cv2.GaussianBlur(img_bak,(3,3),2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t0 = time.time()
            watches = watch_cascade.detectMultiScale(gray, 1.1, 2,minSize=(76 , 18))
            print time.time() - t0
            for (x, y, w, h) in watches:
                #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                patch  = img_bak[y:y + h, x:x + w]
                #cv2.imwrite("./cascade_results/"+str(count)+".png",patch)

                # patch[:,:,0]  = cv2.equalizeHist(patch[:,:,0])
                # patch[:, :, 1] = cv2.equalizeHist(patch[:, :, 1])
                # patch[:, :, 2] = cv2.equalizeHist(patch[:, :, 2])
                vector = tj.norm(patch)
                prob = model.predict(np.array([vector]))[0]
                print prob
                prob_1 = prob[0]
                prob_2 = prob[1]
                #cv2.putText(img,str(round(prob_1,2)),(x,y+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255))




                # if prob_2<0.7 or prob_1>0.5:
                  #  cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
                extend_w = int(w*extend_scale)
                extend_h = int(h*extend_scale)
                start_corr_y = y - int(extend_h*1.0)
                end_coor_y  =y + extend_h + int(extend_h*1.0)
                start_corr_x = x - int(extend_w*0.4)
                end_coor_x = x + extend_w + int(extend_w*0.4)

                if start_corr_x < 0 :
                    start_corr_x = 0;
                if start_corr_y < 0 :
                    start_corr_y = 0;
                if end_coor_x>img.shape[1]:
                    end_coor_x = img.shape[1]
                if end_coor_y>img.shape[0]:
                    end_coor_y = img.shape[0]

                extend_patch = img[start_corr_y:end_coor_y, start_corr_x:end_coor_x]
                cv2.imshow("extend_patch",extend_patch)
                cv2.waitKey(0)


                extend_patch_resize = cv2.resize(extend_patch, (180, 140))

                image = norm(extend_patch_resize)
                vector = models.predict(np.array([image]))
                points  = foutpoint(vector)
                points = resize_with_points(points,extend_patch.shape,extend_patch_resize)
                points = rectSrceen(points,[start_corr_x,start_corr_y])

                remap_points = np.array([[0,0],[136,0],[136,36],[0,36]],dtype=np.float32);

                print points,remap_points
                M = cv2.getPerspectiveTransform(np.array(points,dtype=np.float32),remap_points)
                warp_image  = cv2.warpPerspective(img,M,(136,36))
                cv2.imshow("warp_image",warp_image)
                for one in points:
                    cv2.circle(img, (int(one[0]), int(one[1])), 1, (0, 255, 0), 2)
                    #drawPoints(extend_patch, vector)


                count += 1;





            print count
            cv2.imshow("img",img)
            cv2.waitKey(0)