import cv2
import matplotlib.pyplot as plt
import numpy as np
import Ones
frame=cv2.imread('6.jpg')
frame=cv2.resize(frame,(800,800))
grayscaled=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
kernel = np.ones((3,3),np.float32)/9
dst = cv2.filter2D(grayscaled,-1,kernel)
median=cv2.medianBlur(dst,3)
retval,threshold=cv2.threshold(median,242,255,cv2.THRESH_BINARY)
kernel2=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
kernel2=kernel2*-1
laplacian=cv2.filter2D(threshold,-1,kernel2)
laplacian=255-laplacian
cv2.imwrite('Original_Image.jpg',frame)
cv2.imwrite('Grayscaled_Image.jpg',grayscaled)
cv2.imwrite('Filtering_Smoothing_Image.jpg',threshold)
cv2.imwrite('Boundary_Image.jpg',laplacian)
print('Area:',Ones.Ones(threshold,800,800))
print('Perimeter:',Ones.Ones(laplacian,800,800))
cv2.waitKey(0)    
cv2.destroyAllWindows()
