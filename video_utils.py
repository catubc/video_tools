#video_utils.py

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from sklearn import decomposition

COLORS=['blue','green','violet','lightseagreen','lightsalmon','dodgerblue','mediumvioletred','indianred','lightsalmon','pink','darkolivegreen', 'brown', 'magenta',
'blue','green','violet','lightseagreen','lightsalmon','dodgerblue','mediumvioletred','indianred','lightsalmon','pink','darkolivegreen', 'brown', 'magenta']



def Crop_frame(image, filename, area):

    global coords, image_temp, ax, fig, cid, img_height, img_width
    
    image_temp = image.copy()
    img_height, img_width = image_temp.shape[:2]

    fig, ax = plt.subplots()

    if (os.path.exists(filename[:-4]+area+'_crop.npy')==False):
        coords=[]

        ax.imshow(image)#, vmin=0.0, vmax=0.02)
        ax.set_title("Compute cropped region for: "+area)
        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        #******* MASK AND DISPLAY AREAS OUTSIDE GENERAL MASK 
        #Search points outside and black them out:
        all_points = []
        for i in range(img_height):             #CAN DO THIS FASTER; PYTHONIC
            for j in range(img_width):         
                all_points.append([i,j])

        all_points = np.array(all_points)
        vertixes = np.array(coords) 
        vertixes_path = Path(vertixes)
        print coords
        
        mask = vertixes_path.contains_points(all_points)
        counter=0
        coords_save=[]
        for i in range(img_height):
            for j in range(img_width):
                if mask[counter] == True:
                    image[i][j]=0
                    coords_save.append([i,j])
                counter+=1
        
        print len(coords_save)

        fig, ax = plt.subplots()
        ax.imshow(image)
        plt.show()
       
        np.save(filename[:-4]+area+'_crop', coords_save)

        print "Finished Saving Cropped Pixels"


def Crop_frame_box(image, filename, area):

    global coords, image_temp, ax, fig, cid, img_height, img_width
    
    image_temp = image.copy()
    img_height, img_width = image_temp.shape[:2]

    fig, ax = plt.subplots()

    if (os.path.exists(filename[:-4]+area+'_crop.npy')==False):
        coords=[]

        ax.imshow(image)#, vmin=0.0, vmax=0.02)
        ax.set_title("Compute cropped region for: "+area)
        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        fig, ax = plt.subplots()
        x_coords = np.int32(np.sort([coords[0][0], coords[1][0]]))
        y_coords = np.int32(np.sort([coords[0][1], coords[1][1]]))
        print x_coords, y_coords
        ax.imshow(image[x_coords[0]:x_coords[1], y_coords[0]: y_coords[1]])
        
        coords_out = np.vstack((x_coords, y_coords))
        plt.show()
        
        return coords_out
        
def on_click(event):
    global coords, image_temp, ax, fig, cid
        
    if event.inaxes is not None:
        coords.append((event.ydata, event.xdata))
        for j in range(len(coords)):
            for k in range(3):
                for l in range(3):
                    image_temp[min(img_height,int(coords[j][0])-1+k)][min(img_width,int(coords[j][1])-1+l)]=[0,0,0]

        ax.imshow(image_temp)
        fig.canvas.draw()
    else:
        print 'Exiting'
        plt.close()
        fig.canvas.mpl_disconnect(cid)



def select_pixels(data, generic_coords):        #******************* NOT CURRENTLY USED ****************

    #Make blank mask
    generic_mask_indexes=np.ma.zeros((data.shape), dtype=np.int8)+1
    for i in range(len(generic_coords)):
        generic_mask_indexes[int(generic_coords[i][0])][int(generic_coords[i][1])] = False

    #Apply full mask; probably FASTER METHOD
    #temp_array = np.ma.array(np.zeros((data.shape), dtype=np.float32)+1, mask=True)
    temp_array = np.ma.array(data, mask=generic_mask_indexes)

    #plt.imshow(temp_array)
    #plt.show()

    #temp_array = np.ma.array(np.zeros((len(data),n_pixels,n_pixels),dtype=np.float32), mask=True)
    #for i in range(0, len(data),1):
    #    temp_array[i] = ...
    
    return temp_array
    
        
    
def dimension_reduction(data, method, filename, area):
    
    import sklearn
    from sklearn import metrics, manifold
    
    matrix_in = data
    
    methods = ['MDS', 'tSNE', 'PCA', 'Sammon']    
    
    print "Computing dim reduction, size of array: ", np.array(matrix_in).shape
    
    if method==methods[0]:
        #MDS Method - SMACOF implementation Nelle Varoquaux
        
        if os.path.exists(filename[:-4]+area+'_'+method+'.npy')==False:
            
            print "... MDS-SMACOF..."
            print "... pairwise dist ..."
            dists = metrics.pairwise.pairwise_distances(matrix_in)
            adist = np.array(dists)
            amax = np.amax(adist)
            adist /= amax
            
            print "... computing MDS ..."
            mds_clf = manifold.MDS(n_components=3, metric=True, n_jobs=-1, dissimilarity="precomputed", random_state=6)
            results = mds_clf.fit(adist)
            Y = results.embedding_         
         
            np.save(filename[:-4]+area+'_'+method, Y)
        
        else:
            Y = np.load(filename[:-4]+area+'_'+method+'.npy')


    elif method==methods[1]:
        ##t-Distributed Stochastic Neighbor Embedding; Laurens van der Maaten
        if os.path.exists(filename[:-4]+area+'_'+method+'.npy')==False:


            print "... tSNE ..."
            print "... pairwise dist ..."
            
            dists = sklearn.metrics.pairwise.pairwise_distances(matrix_in)
            
            adist = np.array(dists)
            amax = np.amax(adist)
            adist /= amax
            
            print "... computing tSNE ..."
            model = manifold.TSNE(n_components=3, init='pca', random_state=0)
            Y = model.fit_transform(adist)
            #Y = model.fit(adist)
        
            np.save(filename[:-4]+area+'_'+method, Y)
        
        else:
            Y = np.load(filename[:-4]+area+'_'+method+'.npy')

    elif method==methods[2]:

        if os.path.exists(filename[:-4]+area+'_'+method+'.npy')==False:
            Y = PCA_reduction(matrix_in, 3)
            np.save(filename[:-4]+area+'_'+method, Y)
        else:
            Y = np.load(filename[:-4]+area+'_'+method+'.npy')

                
    #elif method==methods[3]:

        #if os.path.exists(mouse.home_dir+mouse.name+'/tSNE_barnes_hut.npy')==False:
            #print "... computing Barnes-Hut tSNE..."
            #Y = bh_sne(np.array(matrix_in))
        
            #np.save(mouse.home_dir+mouse.name+'/tSNE_barnes_hut', Y)
        #else:
            #Y = np.load(mouse.home_dir+mouse.name+'/tSNE_barnes_hut.npy')

    return Y

def PCA_reduction(X, n_components):


    plt.cla()
    #pca = decomposition.SparsePCA(n_components=3, n_jobs=1)
    pca = decomposition.PCA(n_components=n_components)

    print "... fitting PCA ..."
    pca.fit(X)
    
    print "... pca transform..."
    return pca.transform(X)
        
    
    
    
    
    
