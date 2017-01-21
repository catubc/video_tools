#video_utils.py

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from sklearn import decomposition

COLORS=['blue','green','violet','lightseagreen','lightsalmon','dodgerblue','mediumvioletred','indianred','lightsalmon','pink','darkolivegreen', 'brown', 'magenta',
'blue','green','violet','lightseagreen','lightsalmon','dodgerblue','mediumvioletred','indianred','lightsalmon','pink','darkolivegreen', 'brown', 'magenta']



def cluster_data(cluster_data, cluster_method):
    
    colours = ['blue','red','green','black','orange','magenta','cyan','yellow','brown','pink','blue','red','green','black','orange','magenta','cyan','yellow','brown','pink','blue','red','green','black','orange','magenta','cyan','yellow','brown','pink']
    
    #cluster_method=2
    
    #KMEANS
    if cluster_method == 0: 
        n_clusters = 4
        clusters = cluster.KMeans(n_clusters, max_iter=1000, n_jobs=-1, random_state=1032)
        clusters.fit(cluster_data)

        labels = clusters.labels_

    #MEAN SHIFT
    if cluster_method == 1: 
        from sklearn.cluster import MeanShift, estimate_bandwidth
        from sklearn.datasets.samples_generator import make_blobs
        
        quantile = 0.1
        bandwidth = estimate_bandwidth(cluster_data, quantile=quantile, n_samples=5000)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(cluster_data)
        labels = ms.labels_
        #print labels

    #DBSCAN
    if cluster_method == 2: 
        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.preprocessing import StandardScaler 

        X = StandardScaler().fit_transform(cluster_data)

        eps = 0.2
        
        db = DBSCAN(eps=eps, min_samples=10).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_ 

    #MANUAL
    if cluster_method == 3: 
        manual_cluster(cluster_data)
        
        
    labels = np.array(labels)
    clrs = []
    for k in range(len(labels)):
        clrs.append(colours[labels[k]])
    plt.scatter(cluster_data[:,0], cluster_data[:,1], color=clrs)
    plt.show()
        
#***************************************************************************************
def manual_cluster(data):

    global coords, data_temp, ax, fig, cid
    
    data_temp = data
    
    fig, ax = plt.subplots()
    
    coords=[]
    #ax.imshow(images_processed)#, vmin=0.0, vmax=0.02)
    ax.scatter(data[:,0],data[:,1])
    ax.set_title("Compute generic (outside the brain) mask")
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    cid = fig.canvas.mpl_connect('button_press_event', on_click_single_frame)
    plt.show()

    return

    #******* MASK AND DISPLAY AREAS OUTSIDE GENERAL MASK 
    #Search points outside and black them out:
    all_points = []
    for i in range(len(images_processed)):
        for j in range(len(images_processed)):
            all_points.append([i,j])

    all_points = np.array(all_points)
    vertixes = np.array(coords) 
    vertixes_path = Path(vertixes)
    
    mask = vertixes_path.contains_points(all_points)
    counter=0
    coords_save=[]
    for i in range(len(images_processed)):
        for j in range(len(images_processed)):
            if mask[counter] == False:
                images_processed[i][j]=0
                coords_save.append([i,j])
            counter+=1

    fig, ax = plt.subplots()
    ax.imshow(images_processed)
    plt.show()
   
    genericmask_file = animal.home_dir+animal.name + '/genericmask.txt'
    np.savetxt(genericmask_file, coords_save)

    print "Finished Making General Mask"


#*************************************************************
def on_click_single_frame(event):
    global coords, data_temp, ax, fig, cid
    
    #n_pix = len(images_temp)
    
    print event.inaxes
    
    if event.inaxes is not None:
        coords.append([event.ydata, event.xdata])
        #for j in range(len(coords)):
        #    for k in range(3):
        #        for l in range(3):
        #            images_temp[min(n_pix,int(coords[j][0])-1+k)][min(n_pix,int(coords[j][1])-1+l)]=0

        #ax.imshow(images_temp)
        print coords
        ax.scatter(data_temp[:,0],data_temp[:,1])
        ax.scatter(coords[0][0], coords[0][1], color='red', s=50)
        fig.canvas.draw()
    else:
        print 'Exiting'
        plt.close()
        fig.canvas.mpl_disconnect(cid)
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
        
        if os.path.exists(filename+area+'_'+method+'.npy')==False:
            
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
         
            np.save(filename+area+'_'+method, Y)
        
        else:
            Y = np.load(filename+area+'_'+method+'.npy')


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
        
            np.save(filename+area+'_'+method, Y)
        
        else:
            Y = np.load(filename+area+'_'+method+'.npy')

    elif method==methods[2]:

        if os.path.exists(filename+area+'_'+method+'.npy')==False:
            Y = PCA_reduction(matrix_in, 3)
            np.save(filename+area+'_'+method, Y)
        else:
            Y = np.load(filename+area+'_'+method+'.npy')

                
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
        
    
    
def filter_PCA(X, filtering, plotting): 
    from scipy import signal

    #filtering = False
    #Filter PCA DATA
    if plotting: 
        t = np.linspace(0,len(X[:,0]), len(X[:,0]))/15.
        ax = plt.subplot(3,1,1)
        print len(t), len(X[:,0])
        plt.plot(t, X[:,0])
        plt.title("PCA #1", fontsize = 30)

    x = np.hstack(X[:,0])
    b, a = signal.butter(2, 0.001, 'high') #Implement 2Hz highpass filter
    if filtering: 
        y = signal.filtfilt(b, a, x)
        X[:,0]=y
    
        if plotting: 
            plt.plot(t, y, color='red')

    #*******************
    if plotting: 
        ax = plt.subplot(3,1,2)
        plt.plot(t, X[:,1])
        plt.title("PCA #2", fontsize = 30)

    x = np.hstack(X[:,1])

    if filtering: 
        y = signal.filtfilt(b, a, x)
        X[:,1]=y

        if plotting:
            plt.plot(t, y, color='red')
    

    #******************
    if plotting:
        ax = plt.subplot(3,1,3)
        plt.plot(t, X[:,2])
        plt.title("PCA #3", fontsize = 30)

    x = np.hstack(X[:,2])
    if filtering: 
        y = signal.filtfilt(b, a, x)
        X[:,2]=y
    
        if plotting:
            plt.plot(t, y, color='red')
        
    if plotting:
        plt.show()
        
    
    plt.close()
    
    
    return X 
    
    
