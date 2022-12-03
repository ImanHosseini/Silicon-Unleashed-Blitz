import cv2
from datetime import datetime
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import dis

krnl = [1.0,2.0,1.0,2.0,4.0,2.0,1.0,2.0,1.0]
krnl = [x/16.0 for x in krnl]

def conv(img):
    res = np.zeros(img.shape,dtype=np.float)
    for (i,j),_ in np.ndenumerate(res):
        aux = 0.
        choices = [(i-1,j-1),(i,j-1),(i+1,j-1),(i-1,j),(i,j),(i+1,j),(i-1,j+1),(i,j+1),(i+1,j+1)]
        for i,ch in enumerate(choices):
            x,y = ch
            if y>=img.shape[1] or y<0 or x>=img.shape[0] or x<0:
                continue
            aux += krnl[i]*img[x,y] 
        res[i,j] = aux

def conv_2(img):
    res = gaussian_filter(img,sigma=3)

img = cv2.imread('les.bmp',cv2.IMREAD_GRAYSCALE)
conv(img)
# t0 = datetime.now()
# for i in range(10):
#     conv(img)
# t1 = datetime.now()
# print((t1-t0).microseconds)
# np.__config__.show()
# print(dis.dis(conv))