#compute sift, fast, orb keypoints of a test picture outside T

import os 
import torch 
import cv2
import numpy as np
from matplotlib import pyplot as plt 

print('--- program starts here ---')

T=torch.load("T.pt",weights_only=True)
print("Tensor loaded!")

no_test_person=38 #1 to 38

print('the number of test person is',no_test_person)

path='E:/code/pythoncode/FaceDetection/CroppedYaleOriginal'
filelist=os.listdir(path)
path2=path+'/'+filelist[no_test_person-1]
print('path of folder of test person: ',path2)
filelist2=os.listdir(path2)
#print(filelist2)
#test picture is the 3rd file in folder (position 02)
testpicturepath=path2+'/'+filelist2[2]
print('path of test picture: ',testpicturepath)

testpicture=cv2.imread(testpicturepath,0)
cv2.imshow('test image (original)',testpicture)
cv2.waitKey(500)  

#SIFT = Scale-Invariant Feature Transform
sift = cv2.SIFT_create()
kp_sift = sift.detect(testpicture,None)
img_sift = cv2.drawKeypoints(testpicture, kp_sift, None, color=(255,0,0))
cv2.imshow('test image SIFT',img_sift)
cv2.waitKey(1000)  
 
#FAST = FAST Algorithm for Corner Detection
#FAST = Features from Accelerated Segment Test
fast = cv2.FastFeatureDetector_create()
kp_fast = fast.detect(testpicture,None)
img_fast = cv2.drawKeypoints(testpicture, kp_fast, None, color=(255,0,0))
cv2.imshow('test image ORB',img_fast)
cv2.waitKey(500)  

#ORB = Oriented FAST and Rotated BRIEF
orb = cv2.ORB_create()
kp_orb = orb.detect(testpicture,None)
img_orb = cv2.drawKeypoints(testpicture, kp_orb, None, color=(255,0,0))
cv2.imshow('test image ORB',img_orb)
cv2.waitKey(500)  

print('number of sift keypoints',len(kp_sift))
print('number of fast keypoints',len(kp_fast))
print('number of orb keypoints',len(kp_orb))

cv2.destroyAllWindows()
print('--- program ends here ---')



fig=plt.figure(figsize=(10, 2)) 
rows=1
columns=4
fig.add_subplot(rows, columns, 1) 
plt.imshow(cv2.cvtColor(testpicture, cv2.COLOR_BGR2RGB))
plt.axis('off') 
plt.title("original") 

fig.add_subplot(rows, columns, 2) 
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title("sift keypoints") 

fig.add_subplot(rows, columns, 3) 
plt.imshow(cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title("fast keypoints") 

fig.add_subplot(rows, columns, 4) 
plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title("orb keypoints") 

plt.show() #Display all open figures.
