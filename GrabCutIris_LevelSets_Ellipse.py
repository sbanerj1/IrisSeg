import numpy as np
import cv2
import sys
import time
import morphsnakes
from cv2 import cv
from scipy.misc import imread
from matplotlib import pyplot as ppl
import urllib2
import math
import random
import os
#import cv2.cv as cv

print '*** Iris segmentation using GAC and GrabCut (PSIVT Workshops 2015) ***'
print '*** Authors - Sandipan Banerjee & Domingo Mery ***'
print '*** Usage - python GrabCutIris_LevelSets_Ellipse.py <filename> *** \n'

segF = 'SegResults'
if not os.path.exists(segF):
    os.makedirs(segF)
#f1 = open('resultsFinal.txt','a+')
lvl_left = -1
lvl_right = -1
lvl_up = -1
lvl_down = -1

p_left = -1
p_right = -1
p_up = -1
p_down = -1

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : GREEN, 'val' : 0}
DRAW_FG = {'color' : RED, 'val' : 1}
DRAW_PR_FG = {'color' : BLACK, 'val' : 3}
DRAW_PR_BG = {'color' : WHITE, 'val' : 2}

temp_det = []
lastcx = -1
lastcy = -1
lastr = -1
#print __doc__

# Loading images
if len(sys.argv) == 2:
    filename = sys.argv[1] # for drawing purposes
else:
    print 'Image not found!'

#f1.write(filename)
#f1.write(',')

img = cv2.imread(filename)
img2 = img.copy()                              # copies of the original image
img3 = img.copy()
cimg = img.copy()
#img4 - img.copy()
eyeball_bw = np.zeros(img.shape,np.uint8)
iris_bw = np.zeros(img.shape,np.uint8)
iter = 0
#img2 = img.copy()  

# Stage 1 - Intensity profiling

h,w,d = img.shape
h3 = h/3
w3 = w/3

lft = 1*w3
rt = 2*w3
up = 1*h3
down = 2*h3

hor_l = [0]*(int(down-up)/5 + 1)
ver_l = [0]*(int(rt-lft)/5 + 1)
temp_l = []
hor_list = []
ver_list = []
min_val = 100
ellipse_size = 0
min_x = 0
min_y = 0
maxf = 0
maxs = 0
eoc = 0

i = lft
j = up
while i <= rt:
    j = up
    while j <= down:
        if int(img[j][i][0]) < min_val:
            min_val = int(img[j][i][0])
        j += 1
    i += 1

m = 0
n = up
k = 0
max_blah = 0
while n <= down:
    m = lft
    while m <= rt:
        temp = int(img[n][m][0])
        if temp < (min_val + 20):
            hor_l[k] += 1 
            img3[n][m] = (0,255,0)
            temp_l.append([m,n])
        else:
            img3[n][m] = (255,255,255)
        m += 1
    if hor_l[k] > max_blah:
        max_blah = hor_l[k]
        hor_list = temp_l
    temp_l = []
    n += 5
    k += 1
    
for i in range(len(hor_list)):
    img3[int(hor_list[i][1])][int(hor_list[i][0])] = (0,0,255)

max_t = max_blah

m = 0
n = lft
k = 0
max_blah = 0
temp_l = []
while n <= rt:
    m = up
    while m <= down:
        temp = int(img[m][n][0])
        if temp < (min_val + 20):
            ver_l[k] += 1 
            img3[m][n] = (0,255,0)
            temp_l.append([n,m])
        else:
            img3[m][n] = (255,255,255)
        m += 1
    if ver_l[k] > max_blah:
        max_blah = ver_l[k]
        ver_list = temp_l
    temp_l = []
    n += 5
    k += 1
    
for i in range(len(ver_list)):
    img3[int(ver_list[i][1])][int(ver_list[i][0])] = (255,0,0)
    
if max_blah > max_t:
    max_t = max_blah

cx = 0
cy = 0
hlst = []
vlst = []
sumh = 0
sumv = 0

i = lft

while i <= rt:
    j = up
    while j <= down:
        if int(img[j][i][0]) < (min_val + 20):
            hlst.append(i)
            sumh += i
            vlst.append(j)
            sumv += j
        j += 1
    i += 1

cx = int(sumh/len(hlst))
cy = int(sumv/len(vlst))            
cx1 = 0
cy1 = 0

for i in range(len(hor_list)):
    for j in range(len(ver_list)):
        if (hor_list[i][0] == ver_list[j][0]) and (hor_list[i][1] == ver_list[j][1]):
            cx1 = hor_list[i][0]
            cy1 = hor_list[i][1]
            break
        
img3[cy][cx] = (255,255,255)

# Stage 2 - Contour estimation with GAC

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness
output_file = []
iteration = 1


def contour_iterator(contour):
    while contour:
        yield contour
        contour = contour.h_next()
        
class FitEllipse:

    def __init__(self, source_image, slider_pos):
        self.source_image = source_image
        cv.CreateTrackbar("Threshold", "Result", slider_pos, 255, self.process_image)
        self.process_image(slider_pos)

    def process_image(self, slider_pos):
        global cimg, source_image1, ellipse_size, maxf, maxs, eoc, lastcx,lastcy,lastr
        """
        This function finds contours, draws them and their approximation by ellipses.
        """
        stor = cv.CreateMemStorage()

        # Create the destination images
        cimg = cv.CloneImage(self.source_image)
        cv.Zero(cimg)
        image02 = cv.CloneImage(self.source_image)
        cv.Zero(image02)
        image04 = cv.CreateImage(cv.GetSize(self.source_image), cv.IPL_DEPTH_8U, 3)
        cv.Zero(image04)

        # Threshold the source image. This needful for cv.FindContours().
        cv.Threshold(self.source_image, image02, slider_pos, 255, cv.CV_THRESH_BINARY)

        # Find all contours.
        cont = cv.FindContours(image02,
            stor,
            cv.CV_RETR_LIST,
            cv.CV_CHAIN_APPROX_NONE,
            (0, 0))
        
        maxf = 0
        maxs = 0
        size1 = 0
        
        for c in contour_iterator(cont):
            if len(c) > ellipse_size:
                PointArray2D32f = cv.CreateMat(1, len(c), cv.CV_32FC2)
                for (i, (x, y)) in enumerate(c):
                    PointArray2D32f[0, i] = (x, y)
                    
                
                # Draw the current contour in gray
                gray = cv.CV_RGB(100, 100, 100)
                cv.DrawContours(image04, c, gray, gray,0,1,8,(0,0))
                
                if iter == 0:
                    strng = segF + '/' + 'contour1.png'
                    cv.SaveImage(strng,image04)
                color = (255,255,255)
                
                (center, size, angle) = cv.FitEllipse2(PointArray2D32f)
                
                # Convert ellipse data from float to integer representation.
                center = (cv.Round(center[0]), cv.Round(center[1]))
                size = (cv.Round(size[0] * 0.5), cv.Round(size[1] * 0.5))
                
                if iter == 1:
                    if size[0] > size[1]:
                        size2 = size[0]
                    else:
                        size2 = size[1]
                    
                    if size2 > size1:
                        size1 = size2
                        size3 = size                

                # Fits ellipse to current contour.
                if eoc == 0 and iter == 2:
                    rand_val = abs((lastr - ((size[0]+size[1])/2)))
                    if rand_val > 20 and float(max(size[0],size[1]))/float(min(size[0],size[1])) < 1.5:
                        lastcx = center[0]
                        lastcy = center[1]
                        lastr = (size[0]+size[1])/2
                    
                    if rand_val > 20 and float(max(size[0],size[1]))/float(min(size[0],size[1])) < 1.4:
                        cv.Ellipse(cimg, center, size,
                                  angle, 0, 360,
                                  color,2, cv.CV_AA, 0)
                        cv.Ellipse(source_image1, center, size,
                                  angle, 0, 360,
                                  color,2, cv.CV_AA, 0)               
                
                elif eoc == 1 and iter == 2:
                    (int,cntr,rad) = cv.MinEnclosingCircle(PointArray2D32f)
                    cntr = (cv.Round(cntr[0]), cv.Round(cntr[1]))
                    rad = (cv.Round(rad))
                    if maxf == 0 and maxs == 0:
                        cv.Circle(cimg, cntr, rad, color, 1, cv.CV_AA, shift=0)
                        cv.Circle(source_image1, cntr, rad, color, 2, cv.CV_AA, shift=0)
                        maxf = rad
                    elif (maxf > 0 and maxs == 0) and abs(rad - maxf) > 30:
                        cv.Circle(cimg, cntr, rad, color, 2, cv.CV_AA, shift=0)
                        cv.Circle(source_image1, cntr, rad, color, 2, cv.CV_AA, shift=0)
                        maxs = len(c)                      
        if iter == 1:
            temp3 = 2*abs(size3[1] - size3[0])
            if (temp3 > 40):
                eoc = 1


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[map(slice, shape)].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u

def test_iris():
    global lvl_up,lvl_down,lvl_left,lvl_right
    # Load the image.
    img_lvl = imread(filename)/255.0
    
    # g(I)
    gI = morphsnakes.gborders(img_lvl, alpha=2200, sigma=5.48)
    
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
    mgac.levelset = circle_levelset(img_lvl.shape, (cy, cx), (int(max_t/2) + 30))
    
    # Visual evolution.
    ppl.figure()
    ij = morphsnakes.evolve_visual(mgac, num_iters=120, background=img_lvl)
    #print ij.shape
    
    x_list = []
    y_list = []
    
    for i in range(w-1):
        for j in range(h-1):
            if ij[j][i] == 0:
                eyeball_bw[j][i] = (255,0,0)
            else:
                x_list.append(i)
                y_list.append(j)
                eyeball_bw[j][i] = (0,0,255)
    
    lvl_down = max(y_list)
    lvl_up = min(y_list)
    lvl_right = max(x_list)
    lvl_left = min(x_list)

test_iris()

def test_pupil():
    global p_up,p_down,p_left,p_right
    # Load the image.
    img_lvl = imread(filename)/255.0
    
    # g(I)
    gI = morphsnakes.gborders(img_lvl, alpha=2200, sigma=5.48)
    
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
    mgac.levelset = circle_levelset(img_lvl.shape, (cy, cx), (max_t*0.3))
    
    # Visual evolution.
    ppl.figure()
    ij = morphsnakes.evolve_visual(mgac, num_iters=50, background=img_lvl)
    
    x_list = []
    y_list = []
    
    for i in range(w-1):
        for j in range(h-1):
            if ij[j][i] == 0:
                iris_bw[j][i] = (255,0,0)
            else:
                x_list.append(i)
                y_list.append(j)
                iris_bw[j][i] = (0,0,255)
    
    p_down = max(y_list)
    p_up = min(y_list)
    p_right = max(x_list)
    p_left = min(x_list)

test_pupil()
    
if (p_left - lvl_left) > 1.3*(lvl_right - p_right):
    print 'Left WRONG'
    lvl_left = lvl_left + int((p_left - lvl_left)-(lvl_right - p_right))
elif (lvl_right - p_right) > 1.3*(p_left - lvl_left):
    print 'Right WRONG'
    lvl_right = lvl_right - int((lvl_right - p_right)-(p_left - lvl_left)) 

if (p_right - p_left) > (p_down - p_up):
    ellipse_size = (p_right - p_left)
else:
    ellipse_size = (p_down - p_up)
 
ellipse_size = 2*ellipse_size

# STage 3 - GrabCut

mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
output = np.zeros(img.shape,np.uint8)           # output image to be shown

# input and output windows
cv2.namedWindow('output')
cv2.namedWindow('input')
cv2.moveWindow('input',img.shape[1]+10,90)

rect_over = True
cv2.rectangle(img,(lvl_left,lvl_down),(lvl_right,lvl_up),BLUE,2)
rect = (min(lvl_left,lvl_right),min(lvl_up,lvl_down),abs(lvl_left-lvl_right),abs(lvl_up-lvl_down))
rect_or_mask = 0
bgdmodel = np.zeros((1,65),np.float64)
fgdmodel = np.zeros((1,65),np.float64)
cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
rect_or_mask = 1

diff = p_up - lvl_up

m = p_left - 2
n = p_up - 2
while n > (p_up - 1.8*(diff/5)):
    cv2.circle(img,(m,n),thickness,value['color'],-1)
    cv2.circle(mask,(m,n),thickness,value['val'],-1)
    m -= 1
    n -= 1

m = p_right + 2
n = p_up + 2
while n > (p_up - 1.8*(diff/5)):
    cv2.circle(img,(m,n),thickness,value['color'],-1)
    cv2.circle(mask,(m,n),thickness,value['val'],-1)
    m += 1
    n -= 1


diff = lvl_down - p_down
m = p_left - 2
n = p_down + 2
while n < (p_down + 1.8*(diff/5)):
    cv2.circle(img,(m,n),thickness,value['color'],-1)
    cv2.circle(mask,(m,n),thickness,value['val'],-1)
    m -= 1
    n += 1

m = p_right + 2
n = p_down + 2
while n < (p_down + 1.8*(diff/5)):
    cv2.circle(img,(m,n),thickness,value['color'],-1)
    cv2.circle(mask,(m,n),thickness,value['val'],-1)
    m += 1
    n += 1
    
diff = (p_left - lvl_left)/10
m = p_left - diff
while m > (lvl_left + diff):
    cv2.circle(img,(m,cy),thickness,value['color'],-1)
    cv2.circle(mask,(m,cy),thickness,value['val'],-1)
    m -= 1
    
diff = (lvl_right - p_right)/10
m = p_right + diff
while m < (lvl_right - diff):
    cv2.circle(img,(m,cy),thickness,value['color'],-1)
    cv2.circle(mask,(m,cy),thickness,value['val'],-1)
    m += 1


diff = p_right - p_left
m = p_left + (diff/5)
value = DRAW_BG
while m < (p_left + 4*(diff/5)):
    cv2.circle(img,(m,cy),thickness,value['color'],-1)
    cv2.circle(mask,(m,cy),thickness,value['val'],-1)
    m += 1

tempi = 0

while tempi < 10:
    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)
    cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
    tempi += 1
   
mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
output = cv2.bitwise_and(img2,img2,mask=mask2)

filename1 = filename.split('\\')
jstname = filename1[1].split('.')
strng = segF + '/' + jstname[0] + '_seg.png'
cv2.imwrite(strng,output)

source_image1 = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE) 
source_image = cv.LoadImage(strng, cv.CV_LOAD_IMAGE_GRAYSCALE)

cv.NamedWindow("Result", 1)

# Stage 4 - Ellipse fitting

fe = FitEllipse(source_image, (min_val+20))

tab1 = cv2.imread(strng)
iter = 1
flag_t = 0

if (p_up - lvl_up) < (0.75*(lvl_down - p_down)):
    flag_t = 1
elif (lvl_down - p_down) < (0.75*(p_up - lvl_up)):
    flag_t = 2
    
if flag_t == 1:
    bnd = p_up - 10
    for i in range(w-1):
        for j in range(h-1):
            if j <= bnd and tab1[j][i][0] == 100:
                tab1[j][i] = (0,0,0)
            #if j <= bnd and tab2[j][i][0] == 255:
                #tab2[j][i] = (0,0,0)
elif flag_t == 2:
    bnd = p_down + 10
    for i in range(w-1):
        for j in range(h-1):
            if j >= bnd and tab1[j][i][0] == 100:
                tab1[j][i] = (0,0,0)

cv2.imwrite(strng,tab1)

source_image = cv.LoadImage(strng, cv.CV_LOAD_IMAGE_GRAYSCALE)
source_image1 = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE) 
fe = FitEllipse(source_image, (min_val+20))

iter = 2
source_image = cv.LoadImage(strng, cv.CV_LOAD_IMAGE_GRAYSCALE)
source_image1 = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE) 
fe = FitEllipse(source_image, (min_val+20))

# Saving results

strng1 = segF + '/' + jstname[0] + '_contour.png'
cv.SaveImage(strng1,source_image1)
cimg1 = cv2.imread(strng1)
bar = np.zeros((img.shape[0],5,3),np.uint8)
res = np.hstack((img2,bar,eyeball_bw,bar,iris_bw,bar,img,bar,output,bar,cimg1))
output_file = segF + '/' + jstname[0] + "_grabcut_output.png"
cv2.imwrite(output_file,res)

print 'Done segmenting!!!'
cv2.destroyAllWindows()