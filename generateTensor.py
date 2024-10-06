#.py file to create a 3d Tensor from the face database

import os 
import cv2
import torch 
#os = operating system functions
#cv2 = openCV = open computer vision
#torch = PyTorch = tensor methods"""

print('--- program starts here ---')

persons=4 
poses=5 
#persons = number of persons used from the face database (min=1, max=38)
#poses = number of poeses per person used from the face database (min=1, max=63)
    
originalheight=192
originalwidth=168
#original picture dimensions

factor=3 #3
height=int(originalheight/factor)
width=int(originalwidth/factor)
#set height and width to smaller values (64 and 56)
picturedimensions=height*width

#T=torch.zeros(picturedimensions,persons,poses)
T=torch.zeros(picturedimensions,persons,poses,dtype=torch.uint8)
#creates 3D tensor with 0-only float values
#T dimensions are picture[height x width] x persons x poses

path='E:/code/pythoncode/FaceDetection/CroppedYaleOriginal'
#path = database folder path
#print('folder with all persons:', path)

filelist=os.listdir(path)
#list with all folders in the database folder
#print(filelist)

for i in range (0, persons):
    #print("loading poses of person ",i+1,"...",int((i+1)/persons*100),"%")
    path2=path+'/'+filelist[i]
    #path2=folder with all pictures of a person
    filelist2=os.listdir(path2)
    for j in range (0,poses):

        #print("person ",i+1,"pose ",j+1,":")
        #first pose is 4th file in folder
        imgpath=path2+'/'+filelist2[j+3]

        a=cv2.imread(imgpath,0) #0=read in grayscale
        a=cv2.resize(a, (width,height))
        #cv2.imshow("a",a)
        #cv2.waitKey(500)
        A=torch.tensor(a)
        A=torch.flatten(A)
        T[:,i,j]=A[:]

torch.save(T,"T.pt")
print("Tensor generated and saved!")
print('--- program ends here ---')



#TESTS
print("--- Tests: ---")
#a=last image processed
cv2.imshow("a",a)
cv2.waitKey(1000)
print("a=",a)
print("shape a=",a.shape)
#.shape = rows, colomns
print("type a=",type(a[0,0]))
print("A=",A)
print("type A=",type(A[0]))
print("T[:,-1,-1]=",T[:,-1,-1])






