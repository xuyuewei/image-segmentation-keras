import numpy as np
import cv2 as cv

def cv_from_seg_to_bbox(segment,
                        process_size = 128
                        img_wdepth = 8,
                        kernel_size = (5,5)):
    
    h,w = segment.shape[0],segment.shape[1]
    scale_factor = w//process_size
    res = cv.resize(segment,(process_size, h//scale_factor), interpolation = cv.INTER_LINEAR)
    #change color from RGB to HSV
    hsv_segment = cv.cvtColor(res,cv.COLOR_BGR2HSV)

    hists = []
    masks = []
    kernel = np.ones((5,5),np.uint8)
    img_color_scale = pow(2,img_wdepth)
    bounding_box = []

    hsv_mask[0] = cv.equalizeHist(hsv_mask[0])

    color_space = [[[0,43,46],[10,255,255]],[[156,43,46],[180,255,255]],[[11,43,46],[25,255,255]],[[26,43,46],[34,255,255]],
                   [[35,43,46],[77,255,255]],[[78,43,46],[99,255,255]],[[100,43,46],[124,255,255]],[[125,43,46],[155,255,255]]]
    

    

    color_space = np.array(color_space.extend([[[0,0,221],[180,30,255]],[[0,0,0],[180,255,46]],[[0,0,46],[180,43,220]]])))

    for cs in color_space:
        mask = cv.inRange(hsv_segment, cs[0], cs[1])
        #denoise
        #Morphological opening
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
        #Morphological closing
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        masks.append(mask)
        #find contours
        contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours[1]:
        #(x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
        x,y,w,h = cv.boundingRect(cnt)
        bounding_box.append([x,y,w,h])
            #Draw a diagonal blue line with thickness of 5 px
            #cv.line(img,(0,0),(511,511),(255,0,0),5)
            # Bitwise-AND mask and original image
            #res = cv.bitwise_and(frame,frame, mask= mask)
            
    return bounding_box

