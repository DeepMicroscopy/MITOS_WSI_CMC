from fastai import *
from fastai.vision import *
import numpy as np
def convert_RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).
    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = (I == 0)
    I[mask] = 1
    return np.maximum(-1 * np.log(I / 255), 1e-6)

def get_concentrations(I, stain_matrix, regularizer=0.01):
    """
    Estimate concentration matrix given an image and stain matrix.
    :param I:
    :param stain_matrix:
    :param regularizer:
    :return:
    """
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    HE = stain_matrix.T
    Y = OD.T
    C = np.asarray(np.linalg.lstsq(HE, Y, rcond=1))
    return C[0].T

def normalize(I, sm_source, sm_target, maxC_source,maxC_target):
        source_concentrations = get_concentrations(I, sm_source)

        source_concentrations *= (maxC_target / maxC_source)

        tmp = np.clip(255 * np.exp(-1 * np.dot(source_concentrations, sm_target)),0,255)

        return tmp.reshape(I.shape).astype(np.uint8)

class StainAugmentedImageItemList(ImageList):
    
    def __init__(self, *args, augmented=True, patchsize=128, convert_mode='RGB', after_open:Callable=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.basepath = kwargs['path']
        self.patchsize = (patchsize, patchsize)
        self.augmented = augmented
#        self.convert_mode,self.after_open = convert_mode,after_open
#        self.copy_new += ['convert_mode', 'after_open']
#        self.c,self.sizes = 3,{}
        self.slide_concentrations = pickle.load(open('../slide_concentrations.p','rb'))
        
    def open(self, fn:PathOrStr)->Image:
        filename, uid, x,y,isval = fn.split(':')
        x,y,isval = int(x), int(y), bool(isval)
        reader = openslide.open_slide(filename)
        image = reader.read_region(location=(x,y), size=(512,512), level=0)
        tensor = pil2tensor(image, np.float32)[0:3]
        if isval:  
            image = Image(tensor[:,0:patchSize,0:patchSize]/255.0)
            return image
        else:
            slide_concentrations = self.slide_concentrations
            target = random.choice(list(slide_concentrations.keys()))
            target_stainmatrix = self.slide_concentrations[target]['sm']
            source_stainmatrix = self.slide_concentrations[os.path.basename(filename)]['sm']
            source_maxc = self.slide_concentrations[os.path.basename(filename)]['maxC']
            target_maxc = self.slide_concentrations[target]['maxC']
            augmented = Tensor(normalize(I=tensor, maxC_source=source_maxc, maxC_target=target_maxc, sm_source=source_stainmatrix,sm_target=target_stainmatrix))
            image = Image(augmented[:,0:patchSize,0:patchSize]/255.0)
            return image


