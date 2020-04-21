from fastai.vision import *
from fastai import *
from tqdm import tqdm
import cv2
import openslide
import numpy as np
import os
from lib.nms_WSI import *
from lib.helper import *
import pickle


class WSIImageGetter():
    def __init__(self, filename, basepath='./WSI', width=128, height=128):
        self.slide = openslide.open_slide(basepath+os.sep+filename)
        self.filename = filename
        self.width = width
        self.height = height
    
    def get_patch(self, x,y):
        return self._get_patch(x,y)
    
    def _get_patch(self, x,y):
        return Image(Tensor(np.array(self.slide.read_region(location=(int(x-self.width/2), int(y-self.width/2)), size=(self.width,self.height), level=0))[:,:,0:3]).permute(2,0,1)/255.)


def processSecondStageClassifier(results, basepath='./WSI/', intermediate='res_out.p', modelpath='CellClassifier_128px.pth',nms_threshold=None, class_idx=1, overwrite=True):


    print('Loading',modelpath)
    learn = load_learner(path='./', file=modelpath)

    print('Performing NMS ...')
    results = nms(results, nms_threshold)
    res_out = dict()

    if not overwrite:
        try:
            res_out = pickle.load(open(intermediate,'rb'))
        except:
            pass
    cntr=0
    savekeys = [x for k,x in enumerate(list(results.keys())) if k%10==0]
    #savekeys.append(list(results.keys)[-1])
    print('Saving at: ',savekeys)
    for filename in results:
        print('Processing %s ..'% filename)
        if filename in res_out:
            continue
        res_out[filename] = list()
        ig = WSIImageGetter(filename,basepath=basepath)
        numimages = len(results[filename])
        for x1,y1,x2,y2,cls,p in tqdm(results[filename]):
            center_x = int(0.5*(x1+x2))
            center_y = int(0.5*(y1+y2))
            img = ig.get_patch(center_x,center_y)
            pred = learn.predict(img)
            
            prob_mitosis = pred[2][class_idx]
            res_out[filename] += [[x1,y1,x2,y2,cls, prob_mitosis.numpy()]]
        
        if filename in savekeys: 
            pickle.dump(res_out,open(intermediate,'wb'))
        

    return res_out
