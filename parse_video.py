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

#from opengl_utils import *
from analysis import plot_3D_distribution_new

from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

import matplotlib.gridspec as gridspec

COLORS=['blue','green','violet','lightseagreen','lightsalmon','dodgerblue','mediumvioletred','indianred','lightsalmon','pink','darkolivegreen', 'brown', 'magenta',
'blue','green','violet','lightseagreen','lightsalmon','dodgerblue','mediumvioletred','indianred','lightsalmon','pink','darkolivegreen', 'brown', 'magenta']


#filename = '/media/cat/12TB/in_vivo/tim/yuki/IA1/video_files/IA1pm_Feb9_30Hz.m4v'
filename ='/media/cat/12TB/in_vivo/tim/yuki/AI3/video_files/AI3_2014-10-28 15-26-09.219.wmv'
camera = cv2.VideoCapture(filename)

data_array = []
movie_array = []

frame_count = 0
last_gray = None
if os.path.exists(filename[:-4]+'.npy')==False:
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        # if we are viewing a video and we did not grab a
        # frame, then we have reached the end of the video
        if not grabbed: 
            break

        frame = frame[450:680, 450:700]
        
        #Save cropped raw image into .npy array
        movie_array.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        #plt.imshow(image)
        #plt.show()


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
        

    np.save(filename[:-4], data_array)
    np.save(filename[:-4]+'_movie', movie_array)

else:
    
    data_array = np.load(filename[:-4]+'.npy')
    movie_array = np.load(filename[:-4]+'_movie.npy')




subsampling = False
subsampled_array = []
if subsampling: 
    print "... subsampling ..."
    for k in range(len(data_array)):
        subsampled_array.append(scipy.misc.imresize(data_array[k], 0.1, interp='bilinear', mode=None))


    #ax = plt.subplot(2,1,1)
    #plt.imshow(subsampled_array[100])
    #ax=plt.subplot(2,1,2)
    #plt.imshow(data_array[100])
    #plt.show()

    data_array = np.array(subsampled_array)


#******************* PCA / DIM REDUCTION *****************
np.random.seed(5)

X = []
for k in range(len(data_array)): 
    X.append(np.ravel(data_array[k]))

plt.cla()
#pca = decomposition.SparsePCA(n_components=3, n_jobs=1)
pca = decomposition.PCA(n_components=3)



if os.path.exists(filename[:-4]+'_pca.npy')==False:
    print "... fitting PCA ..."
    pca.fit(X)
    print "... pca transform..."
    X = pca.transform(X)
    
    np.save(filename[:-4]+'_pca', X)

else:
    
    X = np.load(filename[:-4]+'_pca.npy')


#****************** CLUSTERING OF DATA *******************

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


#****************** PLOT CLUSTERS *******************
plotting_mat = False
plotting_3D = True

if plotting_mat: 

    colors = []
    for k in range(len(X)):
        colors.append(COLORS[labels[k]])


    print "... plotting ..."
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = colors, cmap=plt.cm.spectral)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()


#if plotting_3D: 



class Window(QtGui.QMainWindow):

    def __init__(self, X):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        #self.setWindowTitle("OpenNeuron")

        #LOAD CENTRAL WIDGET
        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        #SET DEFAULT WIDGET TO PROCESS
        ophys_widget = EventTriggeredImaging(self, X)
        self.central_widget.addWidget(ophys_widget)  
        self.central_widget.setCurrentWidget(ophys_widget)

        #self.load_widget = Load(self)
        #self.central_widget.addWidget(self.load_widget)
        
        self.show()
 


class EventTriggeredImaging(QtGui.QWidget):
    def __init__(self, parent, X):
        super(EventTriggeredImaging, self).__init__(parent)
        self.parent = parent
        self.X = X
        layout = QtGui.QGridLayout()    
        #plot_3D_distribution_new(self)      #Plot 3D points a
        plot_3D_distribution_new (self)

        self.setLayout(layout)

def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window(X)
    sys.exit(app.exec_())


run()            


print "... done ..."
quit()   
    


#****************** PLOT EXAMPLES AND AVERAGES DATA *******************
 
gs = gridspec.GridSpec(n_clusters, 10)

#[Ca] stack
for k in range(n_clusters):    
    
    #Plot individual frames
    for d in range(4): 
        ax = plt.subplot(gs[k,d])
        plt.imshow(data_array[cluster_indexes[k][d]])
        
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        
    #Plot average frame
    ax = plt.subplot(gs[k,d+1])
    plt.imshow(np.mean(data_array[cluster_indexes[k]], axis=0))
    ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)


    #Plot individual frames
    for p in range(4): 
        ax = plt.subplot(gs[k,p+d+2])
        plt.imshow(movie_array[cluster_indexes[k][p]])
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)

    #Plot average frame
    ax = plt.subplot(gs[k,p+d+3])
    plt.imshow(np.mean(movie_array[cluster_indexes[k]], axis=0))
    ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)


    
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
















