import matplotlib.pyplot as plt
import numpy as np
import cv2, os

#filename = '/media/cat/12TB/in_vivo/tim/yuki/IA1/video_files/IA1pm_Feb9_30Hz.m4v'
filename ='/media/cat/12TB/in_vivo/tim/yuki/AI3/video_files/AI3_2014-10-28 15-26-09.219.wmv'
camera = cv2.VideoCapture(filename)

data_array = []

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

        # define the upper and lower boundaries of the HSV pixel
        # intensities to be considered 'skin'
        rLow = 100 #cv2.getTrackbarPos('R-low', 'images')
        rHigh = 255 #cv2.getTrackbarPos('R-high', 'images')

        lower = np.array([0, 0, rLow], dtype = "uint8")
        upper = np.array([255, 255, rHigh], dtype = "uint8")     

        skinMask = cv2.inRange(frame, lower, upper)

        #plt.imshow(asarray(cv2.GetMat(skinMask)))

        # apply a series of erosions and dilations to the mask
        # using an rectangular kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)

        skin = cv2.bitwise_and(frame, frame, mask = skinMask)

        cv2.imshow('Test', skinMask)
        cv2.waitKey(1)
        data_array.append(cv2.cvtColor(skinMask, cv2.COLOR_GRAY2BGR))

            
        print frame_count; frame_count += 1
        if frame_count==1000: break
        

    np.save(filename[:-4], data_array)

else:
    
    data_array = np.load(filename[:-4]+'.npy')
    

from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = data_array
X = []
for k in range(len(data_array)): 
    X.append(np.ravel(data_array[k]))
#y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
print "... fitting PCA ..."
pca.fit(X)
print "... pca transform..."
X = pca.transform(X)

#for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    #ax.text3D(X[y == label, 0].mean(),
              #X[y == label, 1].mean() + 1.5,
              #X[y == label, 2].mean(), name,
              #horizontalalignment='center',
              #bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
## Reorder the labels to have colors matching the cluster results
#y = np.choose(y, [1, 2, 0]).astype(np.float)

print "... plotting ..."
ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()



