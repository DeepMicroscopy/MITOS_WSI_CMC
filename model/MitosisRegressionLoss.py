from fastai import *
from fastai.vision import *
from torch.autograd import Variable

from lib.object_detection_helper import *

class MitosisRegressionLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.metric_names = ['RegLoss']
       
    def target(self, bbox_tgts,clas_tgts):
        reg_loss = torch.zeros(1).to(clas_tgts.device)#torch.tensor(0, dtype=torch.float32).to(output.device)
        for ct, bt in zip(clas_tgts, bbox_tgts):
            bt = tlbr2cthw(bt)
            area=0
            for singlebox, target in zip(bt,ct):
                
                area += singlebox[3]*singlebox[2]*ct*10
       #     print('est:',re, 'should be:',sum(bt[:,2]*bt[:,3]), 'loss:',((re-sum(bt[:,2]*bt[:,3]))).pow(2))
#            print('Shape of CT:',ct.shape)
            # vermutung: CT: num X 2
#            print('CT:',ct)
#            print('TargeT:', ct[ct>0].sum())
#            print('RE: ',re.view(-1).shape)
#            print('Result:', ((re.sum()-ct[ct>0].sum())/10).pow(2).shape)
            
            reg_loss += area
        return reg_loss
    def forward(self, output, bbox_tgts, clas_tgts):
        reg_estimate = output
   #     print('Estimate:',reg_estimate.shape, bbox_tgts.shape, output.shape)
        reg_loss = torch.zeros(1).to(output.device)#torch.tensor(0, dtype=torch.float32).to(output.device)
        for re, ct, bt in zip(reg_estimate, clas_tgts, bbox_tgts):
            bt = tlbr2cthw(bt)
            area=0
            for singlebox, target in zip(bt,ct):
                
#                print('Target:',target,'Box:', singlebox, 'Area:', singlebox[3]*singlebox[2])
                area += singlebox[3]*singlebox[2]*target*10
       #     print('est:',re, 'should be:',sum(bt[:,2]*bt[:,3]), 'loss:',((re-sum(bt[:,2]*bt[:,3]))).pow(2))
#            print('Shape of CT:',ct.shape)
            # vermutung: CT: num X 2
#            print('CT:',ct)
#            print('TargeT:', ct[ct>0].sum())
#            print('RE: ',re.view(-1).shape)
#            print('Result:', ((re.sum()-ct[ct>0].sum())/10).pow(2).shape)
            
            reg_loss += ((re-area)).pow(2)
        #print('Loss: ', reg_loss[0])
        self.metrics = dict(zip(self.metric_names, [reg_loss[0]/clas_tgts.shape[0]]))
        return reg_loss[0]

