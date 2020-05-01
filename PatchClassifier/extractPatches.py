import numpy as np 
import SlideRunner.general.dependencies
from SlideRunner.dataAccess.database import Database
from SlideRunner.dataAccess.annotations import *
import os
import openslide
import sqlite3
import cv2
import sys

DB = Database()

vp = ViewingProfile()
vp.majorityClassVote=True

cm=np.zeros((7,7))
if len(sys.argv)<2:
    print('Synopsis: ',sys.argv[0],'<valrun>')
    sys.exit()
threshold = 5

disagreedclass = 0
agreedclass = 0
basepath='../WSI/'
patchSize=128

os.system('mkdir -p Data_CMC%s' % sys.argv[1])

dirs = ['Mitosis', 'Nonmitosis']
for k in dirs:
    os.system('mkdir -p Data_CMC%s/train/%s' % (sys.argv[1],k))
    os.system('mkdir -p Data_CMC%s/test/%s' % (sys.argv[1],k))

def listOfSlides(DB):
    DB.execute('SELECT uid,filename from Slides')
    return DB.fetchall()

slidelist_test_1 = ['18','3', '22','10','15','21','14']
slidelist_test_2 = ['1', '20','17','5', '2', '11','16']
slidelist_test_3 = ['12','13','7', '19','8', '6', '9']

test_slides = { '1': slidelist_test_1,
                '2': slidelist_test_2,
                '3': slidelist_test_3}

from os import system
DB.open('../databases/MITOS_WSI_CMC_ODAEL_TR.sqlite')

for slideid,filename in listOfSlides(DB):
    DB.loadIntoMemory(slideid)
    
    
    slide=openslide.open_slide(basepath+filename)

    for k in DB.annotations.keys():

        anno = DB.annotations[k]

        if anno.deleted or anno.annotationType != AnnotationType.SPOT:
            continue
        coord_x = anno.x1
        coord_y = anno.y1

        lu_x = int(coord_x - int(patchSize/2))
        lu_y = int(coord_y - int(patchSize/2))
        img = np.array(slide.read_region(location=(lu_x, lu_y), level=0, size=(patchSize, patchSize)))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


        istest = 'train/' if str(slideid) not in test_slides[sys.argv[1]] else 'test/'
        if (anno.agreedClass ==2):
            cv2.imwrite('images/Mitosis/%d.png' % k, img)
            system(f'ln -s ../../../images/Mitosis/{k}.png Data_CMC{sys.argv[1]}/'+istest+'Mitosis/%d.png' %k)

        if (anno.agreedClass==1):
            cv2.imwrite('images/Nonmitosis/%d.png' % k, img)
            system(f'ln -s ../../../images/Nonmitosis/{k}.png Data_CMC{sys.argv[1]}/'+istest+'Nonmitosis/%d.png' %k)
            




