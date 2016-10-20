import matplotlib.pyplot as plt
import numpy as np
import cv2

filename = "/home/user/project2/vid_files/AI3_2014-10-30 13-23-19.601.npy"

data = np.load(filename)

print data.shape


plt.imshow(data[10,:,:,2])
plt.show()

quit()
#vid_array = np.array(vid_array)

#cv2.imshow("Test", vid_array[50])
#cv2.waitKey(10)
    
#plt.imshow(np.mean(vid_array[50,:,:,:], axis=2))
#plt.imshow(vid_array[50])
#image = cv2.cvtColor(vid_array[50], cv2.COLOR_BGR2RGB)
#plt.imshow(vid_array[50])
#plt.show()

np.save(filename[:-4], vid_array)
#print vid_array.shape
    #cv2.imshow("Test", image)
    #cv2.waitKey(1)
