import sip
sip.setapi('QString', 2) #Sets the qt string to native python strings so can be read without weird stuff

import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib.pyplot as plt
import numpy as np
import cv2, os
import scipy

from openglclasses import *

from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

import matplotlib.gridspec as gridspec

app = QtGui.QApplication(sys.argv)      #***************************** NEED TO CALL THIS AT THE TOP OR OPENGL FUNCTIONS WON"T WORK!!!


COLORS=['blue','green','violet','lightseagreen','lightsalmon','dodgerblue','mediumvioletred','indianred','lightsalmon','pink','darkolivegreen', 'brown', 'magenta',
'blue','green','violet','lightseagreen','lightsalmon','dodgerblue','mediumvioletred','indianred','lightsalmon','pink','darkolivegreen', 'brown', 'magenta']


#filename = '/media/cat/12TB/in_vivo/tim/yuki/IA1/video_files/IA1pm_Feb9_30Hz.m4v'
filename ='/media/cat/12TB/in_vivo/tim/yuki/AI3/video_files/AI3_2014-10-28 15-26-09.219.wmv'
camera = cv2.VideoCapture(filename)

area = '_snout'; 

data_array = []
movie_array = []

frame_count = 0
last_gray = None
if os.path.exists(filename[:-4]+area+'.npy')==False:
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        # if we are viewing a video and we did not grab a
        # frame, then we have reached the end of the video
        if not grabbed: 
            break

        frame = frame[150:400, 750:1080]
        
        #Save cropped raw image into .npy array
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        movie_array.append(image)
        
        #plt.imshow(image)
        #plt.show()

        #quit()

        # define the upper and lower boundaries of the HSV pixel
        # intensities to be considered 'skin'
        rLow = 100 #cv2.getTrackbarPos('R-low', 'images')
        rHigh = 255 #cv2.getTrackbarPos('R-high', 'images')

        lower = np.array([0, 0, rLow], dtype = "uint8")
        upper = np.array([255, 255, rHigh], dtype = "uint8")     

        
        # apply a series of erosions and dilations to the mask
        # using an rectangular kernel
        skinMask = cv2.inRange(frame, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)

        skin = cv2.bitwise_and(frame, frame, mask = skinMask)

        #cv2.imshow('Test', skinMask)
        #cv2.waitKey(1)
        data_array.append(cv2.cvtColor(skinMask, cv2.COLOR_GRAY2BGR))

            
        print frame_count; frame_count += 1
        #if frame_count==1000: break
        

    np.save(filename[:-4]+area, data_array)
    np.save(filename[:-4]+area+'_movie', movie_array)

else:
    
    data_array = np.load(filename[:-4]+area+'.npy')
    movie_array = np.load(filename[:-4]+area+'_movie.npy')


subsampled_array = []
print "... subsampling ..."

if os.path.exists(filename[:-4]+area+'_subsampled.npy')==False:
    for k in range(len(data_array)):
        subsampled_array.append(scipy.misc.imresize(data_array[k], 0.1, interp='bilinear', mode=None))

    data_array = np.array(subsampled_array)

    np.save(filename[:-4]+area+'_subsampled', data_array)
else:
    data_array = np.load(filename[:-4]+area+'_subsampled.npy')



#******************* PCA / DIM REDUCTION *****************
if os.path.exists(filename[:-4]+area+'_pca.npy')==False:
    print "... computing original PCA..."
    np.random.seed(5)

    X = []
    for k in range(len(data_array)): 
        X.append(np.ravel(data_array[k]))

    plt.cla()
    #pca = decomposition.SparsePCA(n_components=3, n_jobs=1)
    pca = decomposition.PCA(n_components=3)

    print "... fitting PCA ..."

    pca.fit(X)
    print "... pca transform..."
    X = pca.transform(X)
    np.save(filename[:-4]+area+'_pca', X)
else:
    X = np.load(filename[:-4]+area+'_pca.npy')


#****************** CLUSTERING OF DATA *******************

if False: 
    def KMEANS(data, n_clusters):

        from sklearn import cluster, datasets
        clusters = cluster.KMeans(n_clusters, max_iter=1000, n_jobs=-1, random_state = 1032)
        clusters.fit(data)
        
        return clusters.labels_

    n_clusters = 20
    labels = KMEANS(X, n_clusters)

    #Save index location for each chosen cluster

    cluster_indexes = []
    for k in range(n_clusters):
        cluster_indexes.append([])

    for p in range(len(labels)):
        cluster_indexes[labels[p]].append(p)


#plotting_mat = False

#if plotting_mat: 

    #colors = []
    #for k in range(len(X)):
        #colors.append(COLORS[labels[k]])


    #print "... plotting ..."
    #fig = plt.figure(1, figsize=(4, 3))
    #plt.clf()
    #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    #ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = colors, cmap=plt.cm.spectral)

    #ax.w_xaxis.set_ticklabels([])
    #ax.w_yaxis.set_ticklabels([])
    #ax.w_zaxis.set_ticklabels([])

    #plt.show()


#****************** PLOT CLUSTERS *******************

plotting_3D = True

if plotting_3D: 
    ctr= 0
    loop_condition = True
    
    filename ='/media/cat/12TB/in_vivo/tim/yuki/AI3/video_files/AI3_2014-10-28 15-26-09.219.wmv'
    PCA_locs = np.load(filename[:-4]+area+'_pca.npy') #[:5000]
    movie_array = np.load(filename[:-4]+area+'_movie.npy')
    #movie_array = movie_array
    ctr+=1

    while loop_condition: 

        #Restart GUI 
        GUI = Window()
        GUI.X = PCA_locs 
        GUI.movie_array = movie_array

        #Show window
        GUI.view_3D()
        app.exec_()
        
        
        #Continue loop if True
        loop_condition = GUI.loop_condition

        #Update data arrays
        print "... picked indexes: ", GUI.saved_indexes
        print "... len cluster: ", len(GUI.saved_indexes)
        data_array = np.delete(data_array, GUI.saved_indexes, axis=0)
        movie_array = np.delete(movie_array, GUI.saved_indexes, axis=0)     #Make sure to udpate the movie arrays also to see correct 
        
        #Redo PCA
        print "... recomputing  PCA..."
        X = []
        for k in range(len(data_array)): X.append(np.ravel(data_array[k]))

        plt.cla()
        pca = decomposition.PCA(n_components=3)

        print "... fitting PCA ..."

        pca.fit(X)
        print "... pca transform..."
        PCA_locs = pca.transform(X)
        #np.save(filename[:-4]+area+'_pca', X)


quit()

