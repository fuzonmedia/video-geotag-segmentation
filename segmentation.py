import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture('video1.mp4')
def k_means(img):    
    #APPLIED K-MEANS CLUSTERING
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2 #K=6
    ret, label, center = cv2.kmeans(Z, K, cv2.KMEANS_USE_INITIAL_LABELS, criteria,5, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    #print("\n centre:", center)
    return res2

def Convert_luv(img):
    #CONVERTED TO A LUV IMAGE AND MADE EMPTY IMAGE, A MASK
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    LUV = cv2.cvtColor(blur, cv2.COLOR_RGB2LUV)
    l = LUV[:, :, 0]
    v1 = l > 80
    v2 = l < 150
    value_final = v1 & v2 
    empty_img[value_final] = 255
    empty_img[LUV[:,:100,:]] = 0   
    #APPLIED BITWISE-AND ON GRAYSCALE IMAGE AND EMPTY IMAGE TO OBTAIN ROAD AND SOME-OTHER IMAGES TOO
    final = cv2.bitwise_and(gray,empty_img)  
    contours, hierchary = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final = cv2.drawContours(final, contours, -1, 0, 3)
    #cv2.imshow("convert_luv", final)
    return final

def filter1(img):
    #APPLIED EROSION,CONTOURS AND TOP-HAT TO REDUCE NOISE
    kernel = np.ones((3,3),np.uint8)
    final_eroded = cv2.erode(img, kernel, iterations=1) 
    contours, hierchary = cv2.findContours(final_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(final_eroded, contours, -1, 0, 3)
    final_waste = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations = 2) 
    final_waste = cv2.bitwise_not(final_waste)
    img = cv2.bitwise_and(final_waste, img)   
    #MADE A LINE ON THE LEFT-BOTTOM OF THE PAGE
    #img = cv2.line(img, (40, height), (1700, height), 255, 100)
    #cv2.imshow("filter1 function output",img)
    #final_masked = cv2.line(final_masked,(width-300,height),(width,height),255,70)
    return img

def shadow(img):
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def floodFill(img):
    #USED FLOOD-FILL TO FILL IN THE SMALL BLACK LANES
    final_flood = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(final_flood,mask,(0,0),255)
    final_flood = cv2.bitwise_not(final_flood)
    final_filled= cv2.bitwise_or(img,final_flood)
    return final_filled

def centeroid(img):
    #gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#converting to grayscale image
    ret,thresh = cv2.threshold(img,127,255,0)#converting to binary image
    M = cv2.moments(thresh)#calculate moments of binary image
    cX = int(M["m10"] / M["m00"])#calculate x coordinate of center
    cY = int(M["m01"] / M["m00"])#calculate y coordinate of center
    return cX,cY

def roi(img):
    black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8) #black in RGB
    black1 = cv2.rectangle(black,(0,208),(1275,715),(255, 255, 255), -1)#the dimension of the ROI
    gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)#converting to gray
    ret,b_mask = cv2.threshold(gray,127,255, 0)#converting to binary image
    fin = cv2.bitwise_and(img,b_mask,mask = b_mask)#masking  the canny and the roi black image
    #cv2.imshow("roi fin",fin)
    return fin

while(cap.isOpened()):
    #frame by frame of video
    ret, image = cap.read()
    image_new = image
       
    #creating empty image of same size
    height, width, no_use = image_new.shape
    empty_img = np.zeros((height, width), np.uint8)  
    res2 = k_means(image_new)   
    final = Convert_luv(res2) 
    shadow_img = shadow(final)  
    final_masked = filter1(final)
    final_filled = floodFill(final_masked)
    #final_filled = roi(final_filled)
    cX,cY = centeroid(final_filled)
    
    s=str(float(cX))+" "+str(float(cY))
   
    #final_blurred = cv2.GaussianBlur(final_filled,(5,5),0)   
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.circle(image, (cX, cY), 2, (0, 255, 0), -1)
    cv2.putText(image, s, (cX - 40, cY + 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, "Centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('original',image)
    cv2.circle(final_filled, (cX, cY), 5, (50, 50, 0), -1)
    cv2.putText(final_filled, s, (cX - 40, cY + 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
    cv2.putText(final_filled, "Centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
    cv2.namedWindow('tried_extraction', cv2.WINDOW_NORMAL)
    cv2.imshow('tried_extraction',final_filled)
    cv2.namedWindow("shadow img", cv2.WINDOW_NORMAL)
    cv2.imshow("shadow img",final_masked)
    #if cv2.waitKey(1) & 0xFF == ord ('k'):
        #cv2.imwrite("segmentation"+str(time.time())+".png",final_filled)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
#cv2.destroyAllWindows()