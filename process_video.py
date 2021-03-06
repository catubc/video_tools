import sip
sip.setapi('QString', 2) #Sets the qt string to native python strings so can be read without weird stuff

import sys
import glob

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib.pyplot as plt
import numpy as np
import cv2, os
import scipy

from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster, datasets

from sklearn import decomposition

import matplotlib.gridspec as gridspec

from openglclasses import *
from video_utils import *

app = QtGui.QApplication(sys.argv)      #***************************** NEED TO CALL THIS AT THE TOP OR OPENGL FUNCTIONS WON"T WORK!!!


#****************************************************************************************
#filename = '/media/cat/12TB/in_vivo/tim/yuki/IA1/video_files/IA1pm_Feb9_30Hz.m4v'
#filename ='/media/cat/12TB/in_vivo/tim/yuki/AI3/video_files/AI3_2014-10-28 15-26-09.219.wmv'
filename = '/media/cat/12TB/in_vivo/tim/yuki/IA1/video_files/IA1am_May10_Week5_30Hz.m4v'

#filename = '/media/cat/All.Data.3TB/in_vivo/tim/yuki/IA2/video_files/IA2pm_Apr21_Week2_30Hz.m4v'


path_dir, fname = os.path.split(filename)
fname = fname[:-4]

areas = ['lever', 'lick', 'snout', 'rightpaw', 'leftpaw', 'grooming'] 


#************************************************************************************************
#*************************************** DEFINE CROPPED AREAS ***********************************
#************************************************************************************************

camera = cv2.VideoCapture(filename)

#Find 200th frame in video: #Save cropped raw image into .npy array
ctr = 0
while ctr<200: 
    (grabbed, frame) = camera.read()
    ctr+=1

image_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image_original_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


#data_norm = ((image_original_gray-np.min(image_original_gray))/(np.max(image_original_gray)-np.min(image_original_gray))*255).astype(np.uint8)      #Normalize data to gray scale 0..255
#img_arctan = np.ma.arctanh((image_original_gray-128.)/128.)
#plt.imshow(img_arctan)
#plt.show()

pathname = path_dir.replace("video_files", "tif_files/")+fname+"/"

for area in areas:
    if os.path.exists(pathname+area+'_crop.npy'): continue

    #Crop_frame(image_original, filename, area)         #DEFINE IRREGULAR CROPPING AREAS
    coords_out = Crop_frame_box(image_original, filename, area)      #DEFINE BOX AREAS FOR CROPPING
    
    np.save(pathname+area+'_crop', coords_out)
    
    print coords_out
    print "Finished Saving Cropped Pixels"


#************************************************************************************************
#************************************* LOAD CROPPED IMAGE STACK *********************************
#************************************************************************************************
#areas = ['lever'] 
#areas = ['lick'] 
areas = ['lever']

for area in areas: 
    if os.path.exists(pathname+area+'_2D.npy')==False:
        
        
        #Reload data from scratch
        #filename = '/media/cat/12TB/in_vivo/tim/yuki/IA1/video_files/IA1am_May10_Week5_30Hz.m4v'
        camera = cv2.VideoCapture(filename)

        path_dir, fname = os.path.split(filename)
        #Find blueLight trigger and only process data during blue light to match other analysis
        print path_dir.replace("video_files","tif_files")+'/'+ filename[:-4]+"/*blue_light_frames.npy"
        blue_light_filename = glob.glob(path_dir.replace("video_files","tif_files")+'/'+ fname[:-4]+"/*blue_light_frames.npy")[0]
        
        blue_light_onoff = np.load(blue_light_filename)
        blue_light_onoff = [blue_light_onoff[0], blue_light_onoff[-1]]  #Set start and end triggers
        

        #Irregular cropping tools
        #generic_mask_file = filename[:-4]+area+'_crop.npy'
        #generic_coords = np.int32(np.load(generic_mask_file))
        #print len(generic_coords)
        
        ##Convert generic_coords into 1D array 
        #mask_3D = np.zeros(image_original_gray.shape, dtype=np.float32)
        #for k in range(len(generic_coords)):
            #mask_3D[generic_coords[k][0], generic_coords[k][1]] = 1
        #mask_1D = np.ravel(mask_3D)

        #Box cropping tools
        
        crop_box = np.load(pathname+area+'_crop.npy')
        print crop_box
        
        data_1D_array = []
        data_2D_array = []
        movie_array = []
        original_movie_array = []   #Save original movie once along the way

        frame_count = 0

        while frame_count<blue_light_onoff[-1]:
            # grab the current frame
            (grabbed, frame) = camera.read()

            # if we are viewing a video and we did not grab a
            # frame, then we have reached the end of the video
            if not grabbed: 
                break

            if frame_count<blue_light_onoff[0]: 
                frame_count+=1
                continue

            original_movie_array.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            

            frame = frame[crop_box[0][0]:crop_box[0][1], crop_box[1][0]:crop_box[1][1]]
            #cv2.imshow('Test', frame)
            #cv2.waitKey(1)
            
            
            #print "... frame.shape: ", frame.shape
            #Save cropped raw image into .npy array
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            movie_array.append(image)
            #plt.imshow(image)
            #plt.show()

            #Save 1D array for processing
            #image_cropped_1D = np.ravel(image)[mask_1D==1]
            #data_1D_array.append(image_cropped_1D)
            
            #Save cropped 2D frames 
            #image_cropped = np.ravel(image)
            #image_cropped[mask_1D==0]=0
            #image_cropped = image_cropped.reshape(image.shape)
            #plt.imshow(image_cropped)
            #plt.show()

            # define the upper and lower boundaries of the HSV pixel
            # intensities to be considered 'skin'
            
            #I don't understand this anymore                    
            rLow = 0 #cv2.getTrackbarPos('R-low', 'images')
            #rHigh = 100 #cv2.getTrackbarPos('R-high', 'images')
            rHigh = 100 #cv2.getTrackbarPos('R-high', 'images')
            lower = np.array([rLow, rLow, rLow], dtype = "uint8")
            upper = np.array([rHigh, rHigh, rHigh], dtype = "uint8")     

            
            # apply a series of erosions and dilations to the mask
            # using an rectangular kernel
            skinMask = cv2.inRange(frame, lower, upper)
            #skinMask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
           
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            skinMask = cv2.erode(skinMask, kernel, iterations = 1)  #1
            skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

            # blur the mask to help remove noise, then apply the
            # mask to the frame
            skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
            #skinMask = cv2.bitwise_and(frame, frame, mask = skinMask)

            #cv2.imshow('Test', skinMask)
            #cv2.waitKey(2)
            
            #data_2D_array.append(cv2.cvtColor(skinMask, cv2.COLOR_GRAY2BGR))
            data_2D_array.append(skinMask)
            
            #plt.imshow(skinMask)
            #plt.show()
                
            print frame_count; frame_count += 1
            #if frame_count==1000: break
            

        np.save(pathname+area+'_2D' , data_2D_array)
        
        if os.path.exists(pathname+area+'_movie.npy')==False:
            print "...saving area movie file..."
            np.save(pathname+area+'_movie', movie_array)
        
        if os.path.exists(pathname+'movie.npy')==False:
            print "...saving whole move file to .npy ..."
            
            np.save(pathname+"movie", original_movie_array)
        
        data_2D_array = np.array(data_2D_array)
    else:
        print "... loading cropped data from disk ..."
        
        data_2D_array = np.load(pathname+area+'_2D.npy')
        movie_array = np.load(pathname+area+'_movie.npy', mmap_mode='c')


print data_2D_array.shape


#******************** SUBSAMPLING ***********************

if os.path.exists(pathname+area+'_subsampled.npy')==False:
    subsampled_array = []
    print "... subsampling ..."    
    for k in range(len(data_2D_array)):
        subsampled_array.append(scipy.misc.imresize(data_2D_array[k], 0.2, interp='bilinear', mode=None))

    data_2D_array = np.array(subsampled_array)
    np.save(pathname+area+'_subsampled', data_2D_array)

else:
    data_2D_array = np.load(pathname+area+'_subsampled.npy')

print data_2D_array.shape

data_array = data_2D_array  #Revert to old nomenclature

#******************* PCA / DIM REDUCTION *****************
methods = ['MDS', 'tSNE', 'PCA', 'Sammon']
method = methods[2]
print "... computing original dim reduction ..."

X = []
for k in range(len(data_2D_array)):
    X.append(np.ravel(data_2D_array[k]))

X = dimension_reduction(X, method, pathname, area)


#****************** FILTER OUT SLOW COMPONENTS OUT OF DATA *******************

if True: 
    X = filter_PCA(X, filtering=True, plotting=True)


#****************** PLOT DISTRIBUTIONS - 2D & CLUSTER THEM *******************

if False: 
    cluster_method = 3
    cluster_data(X, cluster_method)



#****************** PLOT CLUSTERS *******************

plotting_3D = True

if plotting_3D: 
    loop_condition = True
    
    #Load original movies and increase boundaries 
    crop_box = np.load(pathname+area+'_crop.npy')
    enlarge = 25
    for k in range(2): 
        crop_box[k][0] = max(0,crop_box[k][0]-enlarge); crop_box[k][1] = min(crop_box[k][1]+enlarge, image_original.shape[k])
    movie_array = np.load(pathname+'movie.npy', mmap_mode='c')[:, crop_box[0][0]:crop_box[0][1], crop_box[1][0]:crop_box[1][1]]
    #movie_array = np.load(filename[:-4]+area+'_movie.npy', mmap_mode='r+')
    
    #Load PCA locations;
    #xyz_locs = np.load(filename[:-4]+area+'_'+method+'.npy') #*1E4
    X = X*5     #******************3D PLOT SCALING
    
    cumulative_indexes = []
    while loop_condition: 

        #Start/Restart GUI 
        GUI = Window()
        GUI.X = X 
        GUI.movie_array = movie_array

        #Show window
        GUI.view_3D()
        app.exec_()                 #OPENGL Loops exit on certain conditions.
                
        #Continue loop if True
        loop_condition = GUI.loop_condition


        if loop_condition == False: 
            cumulative_indexes.append(np.arange(len(X)))  #SAVE REMAINING FRAMES AT END - MAKE SURE THIS IS CORRECT...
            break

        
        #Update data arrays
        #print "... picked indexes: ", GUI.saved_indexes
        print "... len cluster: ", len(GUI.saved_indexes)
        print GUI.saved_indexes
        cumulative_indexes.append(np.sort(GUI.saved_indexes))
        
        print "... len cumulative_indexes: ", len(cumulative_indexes)
        #data_array = np.delete(data_array, GUI.saved_indexes, axis=0)
        X = np.delete(X, GUI.saved_indexes, axis=0)
        movie_array = np.delete(movie_array, GUI.saved_indexes, axis=0)     #Make sure to udpate the movie arrays also to see correct frames
        
        #Redo PCA
        print "... recomputing  dim reduction..."
        X_temp = []
        for k in range(len(X)): 
            X_temp.append(np.ravel(X[k]))
        
        #Call PCA FUNCTION HERE....Check for existence outside? 
        plt.cla()
        pca = decomposition.PCA(n_components=3)

        print "... fitting PCA ..."
        pca.fit(X_temp)

        print "... pca transform..."
        X = pca.transform(X_temp)

        #X = filter_PCA(X, filtering=False, plotting=False)

        #np.save(filename[:-4]+area+'_pca', X)



#TEST THE EXTRACTED CLUSTERS  
print "... listing cluster indexes..."         
for k in range(len(cumulative_indexes)):
    print len(cumulative_indexes[k])
    print cumulative_indexes[k][:10]


#Load original movies and increase boundaries 
crop_box = np.load(pathname+area+'_crop.npy')
enlarge = 10
for k in range(2): 
    crop_box[k][0] = max(0,crop_box[k][0]-enlarge); crop_box[k][1] = min(crop_box[k][1]+enlarge, image_original.shape[k])
movie_array = np.load(pathname+'movie.npy', mmap_mode='c')[:, crop_box[0][0]:crop_box[0][1], crop_box[1][0]:crop_box[1][1]]
   
#Compute membership in each cluster and save examples to file:
cluster_ids = []
dim=6
frame_indexes = np.arange(len(movie_array))     #Make original indexes and remove them as they are removed from the datasets 
for k in range(len(cumulative_indexes)):
    img_indexes = np.int32(np.random.choice(cumulative_indexes[k], min(len(cumulative_indexes[k]), dim*dim)))   #Chose random examples from cluster
        
    #Plot individual frames
    gs = gridspec.GridSpec(dim,dim)
    for d in range(len(img_indexes)): 
        ax = plt.subplot(gs[int(d/dim), d%dim])
        #temp_img = np.ma.arctanh((movie_array[img_indexes[d]]-128.)/128.)
        plt.imshow(movie_array[img_indexes[d]])#, cmap='Greys_r')
        ax.set_xticks([]); ax.set_yticks([])
        
    plt.suptitle("Cluster: " + str(k+1) + "/" + str(len(cumulative_indexes))+"  # frames: "+str(len(cumulative_indexes[k])), fontsize = 10)
    plt.tight_layout()  
    
    plt.savefig(pathname+area+'_cluster_'+str(k))   # save the figure to file
    plt.close() 


    #Save cluster ids
    print "... frame_indexes: ", frame_indexes[:10], frame_indexes[-10:]
    print "...cmulative_indexes: ", cumulative_indexes[k][:10], cumulative_indexes[k][-10:]
    correct_frame_indexes = frame_indexes[cumulative_indexes[k]]
    print "...correct_frame_indexes: ", correct_frame_indexes[:10], correct_frame_indexes[-10:]
    cluster_ids.append(correct_frame_indexes)

    #Update frame_indexes by deleting cumulative_indexes
    frame_indexes = np.delete(frame_indexes, cumulative_indexes[k])   
    
    print "...len cumulative indexes: ", len(cluster_ids[k])
    print "...len frame_indexes: ", len(frame_indexes)

#Plot examples from each cluster and prompt user for naming
cluster_names = []
dim=6
frame_indexes = np.arange(len(movie_array))     #Make original indexes and remove them as they are removed from the datasets 

for k in range(len(cumulative_indexes)):
    gs = gridspec.GridSpec(dim,dim)
    img_indexes = np.int32(np.random.choice(cluster_ids[k], min(len(cluster_ids[k]), dim*dim)))

    #Plot individual frames
    gs = gridspec.GridSpec(dim,dim)
    for d in range(len(img_indexes)): 
        ax = plt.subplot(gs[int(d/dim), d%dim])
        #temp_img = np.ma.arctanh((movie_array[img_indexes[d]]-128.)/128.)
        plt.imshow(movie_array[img_indexes[d]])#, cmap='Greys_r')
        ax.set_xticks([]); ax.set_yticks([])
        
    plt.suptitle("Cluster: " + str(k+1) + "/" + str(len(cumulative_indexes)), fontsize = 20)
    plt.show()

    #Save cluster ids
    #correct_frame_indexes = frame_indexes[cumulative_indexes[p]]
    #cluster_ids.append(correct_frame_indexes)

    cluster_names.append(raw_input("Cluster name: "))


for k in range(len(cluster_ids)):
    print len(cluster_ids[k])
    print cluster_ids[k][:20]
    
np.savez(pathname+area+'_clusters.npz', cluster_indexes=cluster_ids, cluster_names=cluster_names)




quit()
#****************** AUTOMATIC CLUSTERING OF DATA *******************

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

quit()

