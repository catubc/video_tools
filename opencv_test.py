import matplotlib.pyplot as plt
import numpy as np
import cv2

filename = "/media/cat/12TB/in_vivo/tim/yuki/AI3/video_files/AI3_2014-10-27 15-03-25.546.wmv"
vid = cv2.VideoCapture(filename)

vid_array = []
ctr=0
while True:
    vid.grab()

    retval, image = vid.retrieve()
    print ctr
    ctr+=1
    
    if ctr>100: break
    if not retval:
        break
    
    #cv2.imshow("Test", image)
    #cv2.waitKey(1)
    
    vid_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

vid_array = np.array(vid_array)

#cv2.imshow("Test", vid_array[50])
#cv2.waitKey(10)
    
#plt.imshow(np.mean(vid_array[50,:,:,:], axis=2))
#plt.imshow(vid_array[50])
#image = cv2.cvtColor(vid_array[50], cv2.COLOR_BGR2RGB)
plt.imshow(vid_array[50])
plt.show()

#np.save(filename[:-4], vid_array)
print vid_array.shape
    #cv2.imshow("Test", image)
    #cv2.waitKey(1)
