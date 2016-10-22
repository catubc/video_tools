# USAGE
# python skindetector.py
# python skindetector.py --video video/skin_example.mov

# import the necessary packages
import numpy as np
import argparse
import cv2
import math

# global params
refPt = []
cropping = False

## functions 
def trackbar_callback(x): 
    pass 

def click_and_crop(event, x, y, flags, param): 
    global refPt, cropping
    
    if event == cv2.EVENT_LBUTTONDOWN: 
        refPt = [(x, y)]
        cropping = False
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = True

def disable_click_and_crop(event, x, y, flags, params):
    pass

def choose_roi():
    global refPt
    
    # choose ROI
    # read first frame and define ROI
    (grabbed, frame) = camera.read()
    # OK, assume frame is grabbed for now...
    first = frame.copy()

    cv2.imshow('ROI', first)
    while True: 
        key = cv2.waitKey(1) & 0xFF

        # erase
        if key == ord('e'): 
            first = frame.copy()
        # copy and break the loop
        elif key == ord('c'): 
            break

        # draw rectangle
        if cropping:
            cv2.rectangle(first, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow('ROI', first)

    # draw ROI
    if len(refPt) == 2: 
        roi = first[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow('ROI', roi)

    print refPt[0][0], refPt[0][1]


def tracking(to_log):
    if to_log:
        video_length = np.int64(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        # data: 
        # col1: left paw center x
        # col2: left paw center y
        # col3: left paw farthest pt x
        # col4: left paw farthest pt y
        # col5: right paw center x
        # col6: right paw center y
        # col7: right paw farthest pt x
        # col8: right paw farthest pt y
        data = np.zeros((video_length, 8))
        
    frame_count = -1
    last_gray = None
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        # if we are viewing a video and we did not grab a
        # frame, then we have reached the end of the video
        if args.get("video") and not grabbed:
            break

        frame_count += 1

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #if last_gray is None:
        #    last_gray = gray
        #    continue

        #frame_diff = cv2.absdiff(last_gray, gray)
        #diff_thresh = cv2.threshold(frame_diff, 25, 255, \
        #    cv2.THRESH_BINARY)[1]
        #diff_thresh = cv2.dilate(diff_thresh, None, iterations=2)
        #cv2.imshow('difference', diff_thresh)
        #last_gray = gray
        #img1, cnt1, hr1 = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, \
        #    cv2.CHAIN_APPROX_SIMPLE)
        #for c in cnt1:
        #    if cv2.contourArea(c) < 100: 
        #        continue
        #    (ax, ay, aw, ah) = cv2.boundingRect(c)
        #    cv2.rectangle(frame, (ax, ay), (ax+aw, ay+ah), (0, 255, 0), 2)
        

        # define the upper and lower boundaries of the HSV pixel
        # intensities to be considered 'skin'
        rLow = cv2.getTrackbarPos('R-low', 'images')
        rHigh = cv2.getTrackbarPos('R-high', 'images')

        lower = np.array([0, 0, rLow], dtype = "uint8")
        upper = np.array([255, 255, rHigh], dtype = "uint8")     

        skinMask = cv2.inRange(frame, lower, upper)


        cv2.imshow('Test', skinMask)
        cv2.waitKey(10)

	    # apply a series of erosions and dilations to the mask
	    # using an rectangular kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

	    # blur the mask to help remove noise, then apply the
	    # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)

        # mask on ROI
        roi = skinMask[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        inv_roi = cv2.bitwise_not(roi)

        # cluster the white pixels; make two copies of 'mask', each with one
        # cluster (kmeans)
        pawMask = np.transpose(np.nonzero(inv_roi))

        Z = np.float32(pawMask)
	if len(Z) == 0:
	    continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(Z, 2, None, criteria, 10, flags)
        if (centers[0, 1] > centers[1, 1]): 
            left = Z[labels.ravel()==1]
            right = Z[labels.ravel()==0]
        else: 
            left = Z[labels.ravel()==0]
            right = Z[labels.ravel()==1]
        
        leftIdx = np.int64(left)
        rightIdx = np.int64(right)

        left_inv_roi = inv_roi.copy()
        left_inv_roi[rightIdx[:, 0], rightIdx[:, 1]] = 0
    
        right_inv_roi = inv_roi.copy()
        right_inv_roi[leftIdx[:, 0], leftIdx[:, 1]] = 0    

        roi_c1 = np.zeros(roi.shape, np.uint8)
        left_roi_drawing = cv2.merge((roi_c1, roi_c1, roi_c1))
        right_roi_drawing = cv2.merge((roi_c1, roi_c1, roi_c1))

        # do paw tracking
        for j in [0, 1]: 
            if (j == 0): 
                this_inv_roi = left_inv_roi
                draw_str = 'left'
                drawing = left_roi_drawing
            else:
                this_inv_roi = right_inv_roi
                draw_str = 'right'
                drawing = right_roi_drawing

            img, contours, hierarchy = cv2.findContours(this_inv_roi, \
                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # if any contours can be found
            if (len(contours) > 0):
                max_area = -1
                ci = 0
                for i in range(len(contours)):
                    cnt = contours[i]
                    area = cv2.contourArea(cnt)
                    if(area > max_area): 
                        max_area = area
                        ci = i
                
		cnt = contours[ci]
		# draw contours
		cv2.drawContours(drawing, [cnt], 0, (0, 0, 255), -1)	    

                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 0, 255), 0)
                # draw the min bounding rotated rect
                #rotated_rect = cv2.minAreaRect(cnt)
                #rotated_rect_box = cv2.boxPoints(rotated_rect)
                #rotated_rect_box = np.int0(rotated_rect_box)
                #cv2.drawContours(drawing, [rotated_rect_box], 0, \
                #    (0, 255, 255), 0)

                # convext hull and defects
                hull = cv2.convexHull(cnt)
		cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

                # calculate the mass center
                moments = cv2.moments(cnt)
                if (moments['m00'] != 0): 
                    mass_cx = int(moments['m10']/moments['m00'])
                    mass_cy = int(moments['m01']/moments['m00'])

                    mass_center = (mass_cx, mass_cy)
                    mass_center_ori = (refPt[0][0]+mass_cx, refPt[0][1]+mass_cy)
                    cv2.circle(drawing, mass_center, 3, [255, 255, 255], -1)
                    cv2.circle(skin, mass_center_ori, 4, [255, 255, 255], -1)

                # compute the convexity defects: next two lines changed
                cnt = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
                hull = cv2.convexHull(cnt, returnPoints = False)
                #hull = cv2.convexHull(cnt, returnPoints = False)
                defects = cv2.convexityDefects(cnt, hull)
                count_defects = 0
                max_y = 0 # track farther point the paw reaches
                max_x = 0

                for i in range(defects.shape[0]): 
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
        
                    # test pts
                    if (end[0] == start[0] and end[1] == start[1]): 
                        print "same point!"                    

                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57.3
                    if angle <= 100:
                        count_defects += 1
                        cv2.circle(drawing, far, 3, [0, 0, 255], -1) # convexity
                                                                 # defects
                        cv2.circle(drawing, end, 3, [255, 0, 0], -1) # fingertips
                        if (start[1] > max_y):
                            max_y = start[1]
                            max_x = start[0]
            
                numStr = draw_str + " number of defects: %d" % count_defects
                cv2.putText(drawing, numStr, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, \
                    0.3, (255, 255, 255), 1) 

                # center of the bounding rect and farthest pt
                center_x = refPt[0][0] + x + w/2    
                center_y = refPt[0][1] + y + h/2                
                farthest_x = refPt[0][0] + max_x                
                farthest_y = refPt[0][1] + max_y
                
                # record data
                if to_log:
                    data[frame_count, j*4] = center_x
                    data[frame_count, j*4+1] = center_y
                    data[frame_count, j*4+2] = farthest_x
                    data[frame_count, j*4+3] = farthest_y       
              
                # draw centers to frame    
                cv2.circle(skin, (center_x, center_y), 4, [0, 255, 0], 2)           
                cv2.circle(skin, (farthest_x, farthest_y), 4, [255, 0, 0], -1)
                cv2.circle(drawing, (x+w/2, y+h/2), 3, [0, 255, 0], 2)
            else: # cannot find contours
                print "cannot find contours from " + draw_str + \
                    " image at frame %d" % frame_count 
                    

	    # show the skin in the image along with the mask
        cv2.imshow("images", np.hstack([frame, skin]))
        cv2.imshow('mask', skinMask)
        cv2.imshow('ROI', roi)
        cv2.imshow('tracking', np.hstack([left_roi_drawing, right_roi_drawing]))

	    # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # save data
    if to_log:
        np.savetxt('results.txt', data, fmt='%.4f')

## main
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())
    
#open windows for image viewing
cv2.namedWindow('ROI', 0) # show mask (ROI)
cv2.resizeWindow('ROI', 512, 384)
cv2.setMouseCallback('ROI', click_and_crop)

## open video    
#if not args.get("video", False):
    #camera = cv2.VideoCapture(0)

## otherwise, load the video
#else:
    #camera = cv2.VideoCapture(args["video"])

filename = '/media/cat/12TB/in_vivo/tim/yuki/IA1/video_files/IA1pm_Feb9_30Hz.m4v'
camera = cv2.VideoCapture(filename)


# choose ROI
#choose_roi()
refPt.append([100, 250])
refPt.append([50, 250])
#click_and_crop


#cv2.setMouseCallback('ROI', disable_click_and_crop)

# open more windows
#cv2.namedWindow('images', 0)
#cv2.resizeWindow('images', 1024, 384)
#cv2.createTrackbar('R-low', 'images', 0, 255, trackbar_callback)
#cv2.createTrackbar('R-high', 'images', 0, 255, trackbar_callback)
#cv2.namedWindow('mask', 0)
#cv2.resizeWindow('mask', 512, 384) # show mask (full frame)
#cv2.namedWindow('tracking', 0) # show tracking 
#cv2.resizeWindow('tracking', 1024, 384)

# choose threshold on red channel
tracking(to_log = False)

# perform tracking
#camera.release()
# open video    
#if not args.get("video", False):
#    camera = cv2.VideoCapture(0)
#else:
#    camera = cv2.VideoCapture(args["video"])






tracking(to_log = True)
camera.release()
cv2.destroyAllWindows()
