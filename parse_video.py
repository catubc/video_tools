import matplotlib.pyplot as plt
import numpy as np
import cv2

filename = "/home/user/project2/vid_files/AI3_2014-10-30 13-23-19.601.wmv"
vid = cv2.VideoCapture(filename)

vid.set(cv2.CAP_PROP_POS_FRAMES, 9500)

vid_array = []
ctr=0
while True:
    #vid.grab()

    (grabbed, frame0) = vid.read()
    if not grabbed: 
	break

    frame_cropped = frame0[400:600, 300:700] # change the ROI here

    cv2.imwrite('img.png', frame_cropped)
    
    cv2.imshow('Test', frame_cropped)
    cv2.waitKey(10)

    #retval, image = vid.retrieve()
    print ctr
    ctr+=1
    
    if ctr>0: break
    #if not retval:
    #    break
    
    #cv2.imshow("Test", image)
    #cv2.waitKey(1)
    
    vid_array.append(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))

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
