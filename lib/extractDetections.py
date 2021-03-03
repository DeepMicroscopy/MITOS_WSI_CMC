import bz2
import pickle
import numpy as np
from sklearn.neighbors import KDTree
import openslide 
from SlideRunner_dataAccess.database import Database
from lib.nms_WSI import nms
import cv2

def _F1_core_enh(centers_DB : np.ndarray, boxes : np.ndarray, score : np.ndarray, det_thres:float):

        to_keep = score>det_thres
        boxes = boxes[to_keep]


        if boxes.shape[-1]==6:
            # 4-coordinates -> x1,y1,x2,y2 --> calculate centers
            center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
            center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
        else:
            center_x = boxes[:, 0] 
            center_y = boxes[:, 1] 

        isDet = np.zeros(boxes.shape[0]+centers_DB.shape[0])
        isDet[0:boxes.shape[0]]=1 # mark as detection, rest ist GT

        if (centers_DB.shape[0]>0):
            center_x = np.hstack((center_x, centers_DB[:,0]))
            center_y = np.hstack((center_y, centers_DB[:,1]))
        radius=25
        

        # set up kdtree 
        X = np.dstack((center_x, center_y))[0]

        if (X.shape[0]==0):
            return 0,0,0,0,[],[],[]

        
        try:
            tree = KDTree(X)
        except:
            print('Shapes of X: ',X.shape)
            raise Error()

        ind = tree.query_radius(X, r=radius)

        annotationWasDetected = {x: 0 for x in np.where(isDet==0)[0]}
        DetectionMatchesAnnotation = {x: 0 for x in np.where(isDet==1)[0]}

        # check: already used results
        alreadyused=[]
        for i in ind:
            if len(i)==0:
                continue
            if np.any(isDet[i]) and np.any(isDet[i]==0):
                # at least 1 detection and 1 non-detection --> count all as hits
                for j in range(len(i)):
                    if not isDet[i][j]: # is annotation, that was detected
                        if i[j] not in annotationWasDetected:
                            print('Missing key ',j, 'in annotationWasDetected')
                            raise ValueError('Ijks')
                        annotationWasDetected[i[j]] = 1
                    else:
                        if i[j] not in DetectionMatchesAnnotation:
                            print('Missing key ',j, 'in DetectionMatchesAnnotation')
                            raise ValueError('Ijks')

                        DetectionMatchesAnnotation[i[j]] = 1

        TP = np.sum([annotationWasDetected[x]==1 for x in annotationWasDetected.keys()])
        FN = np.sum([annotationWasDetected[x]==0 for x in annotationWasDetected.keys()])

        FP = np.sum([DetectionMatchesAnnotation[x]==0 for x in DetectionMatchesAnnotation.keys()])
        F1 = 2*TP/(2*TP + FP + FN)
#        print('X is: ',X.shape,X,isDet)
        TParr = [x for x in annotationWasDetected.keys() if annotationWasDetected[x]==1]
#        print('Annowasdet: ',annotationWasDetected,TParr)
#        print('DetectionMatchesAnnotation',DetectionMatchesAnnotation)
        TPs = np.array(X)[TParr]
        FNs = np.array(X)[[x for x in annotationWasDetected.keys() if annotationWasDetected[x]==0]]
        FPs = np.array(X)[[x for x in DetectionMatchesAnnotation.keys() if DetectionMatchesAnnotation[x]==0]]
        assert(len(FNs)==FN)
        assert(len(FPs)==FP)
        assert(len(TPs)==TP)
                         

        return F1, TP, FP, FN, TPs, FNs, FPs 


def extractPatch(slide, coord_x, coord_y, filename, patchSize=128):
        lu_x = int(coord_x - int(patchSize/2))
        lu_y = int(coord_y - int(patchSize/2))
        img = np.array(slide.read_region(location=(lu_x, lu_y), level=0, size=(patchSize, patchSize)))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(filename, img)


def extractDetections(databasefile, slidepath='', result_boxes=None, resfile=None, det_thres=0.5, hotclass=2,verbose=False):

    
    DB = Database()
    DB = DB.open(databasefile)

    if (result_boxes is None):
        if resfile is None:
            raise ValueError('At least one of resfile/result_boxes must be given')
    
    if (resfile[-3:] == 'bz2'):
        f = bz2.BZ2File(resfile, 'rb')
    else:
        f = open(resfile,'rb')

    result_boxes = pickle.load(f)

    sTP, sFN, sFP = 0,0,0
    F1dict = dict()
    sP = 0
    
    result_boxes = nms(result_boxes, det_thres)
    
    print('Calculating F1 for test set of %d files' % len(result_boxes))
    mitcount = DB.execute(f'SELECT COUNT(*) FROM Annotations where agreedClass={hotclass}').fetchall()
    print('Official count of mitotic figures in DB: ', mitcount)
    
    slideids = []
    
    for resfile in result_boxes:
        boxes = np.array(result_boxes[resfile])
        

        TP, FP, FN,F1 = 0,0,0,0
        slide_id=DB.findSlideWithFilename(resfile,'')
        slideids.append(str(slide_id))
        DB.loadIntoMemory(slide_id)

        annoList=[]
        for annoI in DB.annotations:
            anno = DB.annotations[annoI]
            if anno.agreedClass==hotclass:
                annoList.append([anno.x1,anno.y1])

        centers_DB = np.array(annoList)

        if boxes.shape[0]>0:
            score = boxes[:,-1]
#            print('ID:',resfile,DB.findSlideWithFilename(resfile,''))
            
            F1,TP,FP,FN,TPs, FNs, FPs = _F1_core_enh(centers_DB, boxes, score,det_thres)
        
            slide = openslide.open_slide(slidepath+resfile)
            idx=0
            for singleTP in TPs:
                extractPatch(slide, *singleTP, filename=f'TP/{slide_id}_{singleTP[0]}_{singleTP[1]}.png')
            for singleFP in FPs:
                extractPatch(slide, *singleFP, filename=f'FP/{slide_id}_{singleFP[0]}_{singleFP[1]}.png')
            for singleFN in FNs:
#                print(f'Writing out: FN/{slide_id}_{singleFN[0]}_{singleFN[1]}.png')
                extractPatch(slide, *singleFN, filename=f'FN/{slide_id}_{singleFN[0]}_{singleFN[1]}.png')
            
            print('FN:', FN, '->', FNs)
            if (centers_DB.shape[0] != TP+FN):
                print(resfile,centers_DB.shape[0],TP+FN)
        else: # no detections --> missed all
            FN = centers_DB.shape[0] 
        
        if (verbose):
            print(f'{resfile}: F1:{F1}, TP:{TP}, FP:{FP}, FN:{FN}')


        sTP+=TP
        sFP+=FP
        sP += centers_DB.shape[0]
        sFN+=FN
        F1dict[resfile]=F1
        
    print('Overall: ')
    sF1 = 2*sTP/(2*sTP + sFP + sFN)
    print('TP:', sTP, 'FP:', sFP,'FN: ',sFN,'F1:',sF1)
    print('Number of mitotic figures:',sP)
    print('Precision: .%.3f '%(sTP / (sTP+sFP)))
    print('Recall: %.3f' %(sTP / (sTP+sFN)))
    
    #print('Not working on: ',np.array(DB.execute(f'SELECT uid from Slides where uid not in ({",".join(slideids)})').fetchall()).flatten())

    return sF1, F1dict


"""

    Extraction of detections in the same format as in the TUPAC16 challenge.

    Marc Aubreville, Pattern Recognition Lab, FAU Erlangen-Nürnberg
    
    To reduce complexity of calculation, the F1 score is here derived by querying a KD Tree.
    
    This is possible, since all objects are round and have the same diameter. 

"""


import bz2
import pickle
import numpy as np
from sklearn.neighbors import KDTree
import openslide 
from SlideRunner.dataAccess.database import Database
from lib.nms_WSI import nms
import cv2

def _F1_core_enh(centers_DB : np.ndarray, boxes : np.ndarray, score : np.ndarray, det_thres:float):

        to_keep = score>det_thres
        boxes = boxes[to_keep]


        if boxes.shape[-1]==6:
            # 4-coordinates -> x1,y1,x2,y2 --> calculate centers
            center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
            center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
        else:
            center_x = boxes[:, 0] 
            center_y = boxes[:, 1] 

        isDet = np.zeros(boxes.shape[0]+centers_DB.shape[0])
        isDet[0:boxes.shape[0]]=1 # mark as detection, rest ist GT

        if (centers_DB.shape[0]>0):
            center_x = np.hstack((center_x, centers_DB[:,0]))
            center_y = np.hstack((center_y, centers_DB[:,1]))
        radius=25
        

        # set up kdtree 
        X = np.dstack((center_x, center_y))[0]

        if (X.shape[0]==0):
            return 0,0,0,0,[],[],[]

        
        try:
            tree = KDTree(X)
        except:
            print('Shapes of X: ',X.shape)
            raise Error()

        ind = tree.query_radius(X, r=radius)

        annotationWasDetected = {x: 0 for x in np.where(isDet==0)[0]}
        DetectionMatchesAnnotation = {x: 0 for x in np.where(isDet==1)[0]}

        # check: already used results
        alreadyused=[]
        for i in ind:
            if len(i)==0:
                continue
            if np.any(isDet[i]) and np.any(isDet[i]==0):
                # at least 1 detection and 1 non-detection --> count all as hits
                for j in range(len(i)):
                    if not isDet[i][j]: # is annotation, that was detected
                        if i[j] not in annotationWasDetected:
                            print('Missing key ',j, 'in annotationWasDetected')
                            raise ValueError('Ijks')
                        annotationWasDetected[i[j]] = 1
                    else:
                        if i[j] not in DetectionMatchesAnnotation:
                            print('Missing key ',j, 'in DetectionMatchesAnnotation')
                            raise ValueError('Ijks')

                        DetectionMatchesAnnotation[i[j]] = 1

        TP = np.sum([annotationWasDetected[x]==1 for x in annotationWasDetected.keys()])
        FN = np.sum([annotationWasDetected[x]==0 for x in annotationWasDetected.keys()])

        FP = np.sum([DetectionMatchesAnnotation[x]==0 for x in DetectionMatchesAnnotation.keys()])
        F1 = 2*TP/(2*TP + FP + FN)
#        print('X is: ',X.shape,X,isDet)
        TParr = [x for x in annotationWasDetected.keys() if annotationWasDetected[x]==1]
#        print('Annowasdet: ',annotationWasDetected,TParr)
#        print('DetectionMatchesAnnotation',DetectionMatchesAnnotation)
        TPs = np.array(X)[TParr]
        FNs = np.array(X)[[x for x in annotationWasDetected.keys() if annotationWasDetected[x]==0]]
        FPs = np.array(X)[[x for x in DetectionMatchesAnnotation.keys() if DetectionMatchesAnnotation[x]==0]]
        assert(len(FNs)==FN)
        assert(len(FPs)==FP)
        assert(len(TPs)==TP)
                         

        return F1, TP, FP, FN, TPs, FNs, FPs 


def extractPatch(slide, coord_x, coord_y, filename, patchSize=128):
        lu_x = int(coord_x - int(patchSize/2))
        lu_y = int(coord_y - int(patchSize/2))
        img = np.array(slide.read_region(location=(lu_x, lu_y), level=0, size=(patchSize, patchSize)))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(filename, img)

import os
from lib.nms_WSI import nms
def exportCSV(filepath,resultsfile, threshold):
    results = pickle.load(open(resultsfile,'rb'))
    results = nms(results, threshold)
    for k in results:
        dirname = '%02d' % (int(k.split('_')[0])-73)
        os.system(f'mkdir -p {filepath}/{dirname}')
        boxes = np.array(results[k])
        center_x = (boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2).tolist() if boxes.shape[0]>0 else []
        center_y = (boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2).tolist() if boxes.shape[0]>0 else []
        scores = (boxes[:,-1]).tolist() if boxes.shape[0]>0 else []
        f = open(f'{filepath}/{dirname}/01.csv','w')
        for (cx,cy,s) in zip(center_x,center_y,scores):
            if (s>threshold):
                f.write(f'{int(cy)},{int(cx)}\n')
        f.close()        
