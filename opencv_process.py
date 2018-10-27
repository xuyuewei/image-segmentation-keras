def cv_from_seg_to_bbox():
    hists = []
    img_wdepth = 8
    grain_size = 400
    num_of_bins = 32
    color_scale = pow(2,img_wdepth)
    bin_range = color_scale//num_of_bins
    bounding_box = []
    #change color from RGB to HSV
    hsv_mask = cv.cvtColor(mask,cv.COLOR_BGR2HSV)

    #caculate the color range of mask

    for i in range(3):
        hists.append(cv.calcHist([hsv_mask],[i],None,[num_of_bins],[0,256]))


    color_space = [[i*bin_range,i*bin_range+bin_range] for i in range(num_of_bins) if hists[0][i] > grain_size]

    if hists[1][0] > grain_size:
        color_space.appebd([0,bin_range])

    if hists[2][0] > grain_size:
        color_space.append([0,bin_range])
    
    mask = []
    kernel = np.ones((5,5),np.uint8)

    for cs in color_space:
        mask = cv.inRange(hsv, cs[0], cs[1])
        #denoise
        #Morphological opening
        opening_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        #Morphological closing
        closing_mask = cv.morphologyEx(opening_mask, cv.MORPH_CLOSE, kernel)
        #find contours
        contours, hierarchy = cv.findContours(closing_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
        for cnt in contours:
            leftmost = cnt[cnt[:,:,0].argmin()][0][0]
            rightmost = cnt[cnt[:,:,0].argmax()][0][0]
            topmost = cnt[cnt[:,:,1].argmin()][0][1]
            bottommost = cnt[cnt[:,:,1].argmax()][0][1]
            if ((rightmost-rightmost)>grain_size//20)or((bottommost-topmost)> grain_size//20):
                bounding_box.append([leftmost,topmost,rightmost,bottommost])
            #Draw a diagonal blue line with thickness of 5 px
            #cv.line(img,(0,0),(511,511),(255,0,0),5)
            
