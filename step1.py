
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    #starts -= 1
    ends = starts + lengths -1 
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
    for s, e in zip(starts, ends):
        img[s:e] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
    #return img.reshape(shape)  # Needed to align to RLE direction


# https://www.kaggle.com/iafoss/unet34-dice-0-87/notebook

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
# Data
# In [2]:
PATH = '/home/linsam/github/AirbusShipDetectionChallenge/'
TRAIN = os.path.join(PATH, 'train')
TEST = os.path.join(PATH, 'test')
SEGMENTATION = PATH + '/train_ship_segmentations.csv'

PRETRAINED = PATH + 'models/Resnet34_lable_256_1.h5'
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images
#In [3]:
nw = 2   #number of workers for data loader
arch = resnet34 #specify target architecture
#In [4]:
train_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
for el in exclude_list:
    if(el in train_names): train_names.remove(el)
    if(el in test_names): test_names.remove(el)
#5% of data in the validation set is sufficient for model evaluation
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)
segmentation_df = pd.read_csv(os.path.join(PATH, SEGMENTATION)).set_index('ImageId')



#One of the challenges of this competition is strong data unbalance. Even if only images with ships are considered, the ratio of mask pixels to the total number of pixels is ~1:1000. If images with no ships are included, this ratio goes to ~1:10000, which is quite tough to handle. Therefore, I drop all images without ships, that makes the training set more balanced and also reduces the time per each epoch almost by 4 times. In an independent run, when the dice of my model reached 0.895, I ran it on images without ships and identified ~3600 false positive predictions out ~70k images. The incorrectly predicted images were incorporated to the training set as negative examples, and training was continued. The problem of false positive predictions can be further mitigated by stacking U-net model with a classification model predicting if ships are present in a particular image (https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection - ~98% accuracy). I also noticed that in some kernels the dataset is tried to be balanced by keeping approximately the same number of images with 0, 1, 2, etc. ships. However, this strategy would be effective for such task as ship counting rather than training U-net or SSD. One possible way to balance the dataset is creative cropping the images that keeps approximately the same number of pixels corresponding to a ship or something else. However, I doubt that such approach will effective in this competition. Therefore, a special loss function must be used to mitigate the data unbalance.

#In [5]:
def cut_empty(names):
    return [name for name in names 
            if(type(segmentation_df.loc[name]['EncodedPixels']) != float)]

tr_n = cut_empty(tr_n)
val_n = cut_empty(val_n)
#In [6]:
def get_mask(img_id, df):
    shape = (768,768)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return img.reshape(shape)
    if(type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
    return img.reshape(shape).T
#In [7]:
class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_y(self, i):
        mask = np.zeros((768,768), dtype=np.uint8) if (self.path == TEST) \
            else get_mask(self.fnames[i], self.segmentation_df)
        img = Image.fromarray(mask).resize((self.sz, self.sz)).convert('RGB')
        return np.array(img).astype(np.float32)
    
    def get_c(self): return 0
#The carrently availible on kaggle version of fastai has a bug in RandomLighting data agmentation class. It would be nice if kaggle updated fastai version to the last one, where this and other bugs are fixed.

#In [8]:
class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  #add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x
#In [9]:
def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(20, tfm_y = TfmType.CLASS),
                RandomDihedral(tfm_y = TfmType.CLASS),
                RandomLighting(0.05, 0.05, tfm_y = TfmType.CLASS)]
    tfms = tfms_from_model(arch, sz, crop_type = CropType.NO, tfm_y=TfmType.CLASS, 
                aug_tfms=aug_tfms)
    tr_names = tr_n if (len(tr_n)%bs == 0) else tr_n[:-(len(tr_n)%bs)] #cut incomplete batch
    ds = ImageData.get_ds(pdFilesDataset, (tr_names,TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    #md.is_multi = False
    # md.
    return md



cut,lr_cut = model_meta[arch]
#In [11]:
def get_base():                   #load ResNet34 model
    layers = cut_model(arch(True), cut)
    return nn.Sequential(*layers)

def load_pretrained(model, path): #load a model pretrained on ship/no-ship classification
    weights = torch.load(PRETRAINED, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights, strict=False)
            
    return model
#In [12]:
class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
    
class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
        
    def forward(self,x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()
            
class UnetModel():
    def __init__(self,model,name='Unet'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]
    
    
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
#In [14]:
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()
#In [15]:
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()
#In [16]:
def dice(pred, targs):
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)
#Training
#In [17]:
m_base = load_pretrained(get_base(),PRETRAINED)
m = to_gpu(Unet34(m_base))
models = UnetModel(m)

#models.model

'''
sz = 256 #image size
bs = 2  #batch size

md = get_data(sz,bs)

#In [20]:
learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam

learn.crit = MixedLoss(10.0, 2.0)
learn.metrics=[accuracy_thresh(0.5),dice,IoU]
'''
wd = 1e-7
lr = 1e-2
#In [21]:
learn.freeze_to(1)

'''
learn.fit(lr,1,wds=wd,cycle_len=1,use_clr=(5,8))


learn.save('Unet34_256_0')
'''
#-------------------------------------------------
lrs = np.array([lr/100,lr/10,lr])

learn.unfreeze() #unfreeze the encoder
learn.bn_freeze(True)
'''
#In [25]:
learn.fit(lrs,2,wds=wd,cycle_len=1,use_clr=(20,8))

learn.fit(lrs/3,2,wds=wd,cycle_len=2,use_clr=(20,8))

learn.sched.plot_lr()

learn.save('Unet34_256_1')
'''

#Visualization
#In [29]:
def Show_images(x,yp,yt):
    columns = 3
    rows = min(bs,8)
    fig=plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        fig.add_subplot(rows, columns, 3*i+1)
        plt.axis('off')
        plt.imshow(x[i])
        fig.add_subplot(rows, columns, 3*i+2)
        plt.axis('off')
        plt.imshow(yp[i])
        fig.add_subplot(rows, columns, 3*i+3)
        plt.axis('off')
        plt.imshow(yt[i])
    plt.show()
'''
#In [30]:
learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))
#In [31]:
Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)

sz = 384 #image size
bs = 4  #batch size



md = get_data(sz,bs)

learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam

learn.crit = MixedLoss(10.0, 2.0)
learn.metrics=[accuracy_thresh(0.5),dice,IoU]

learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)

learn.fit(lrs/5,1,wds=wd,cycle_len=2,use_clr=(10,8))


learn.save('Unet34_384_1')


#Visualization
#In [35]:
learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))
#In [36]:
Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)

'''

#Training (768x768)
#In [37]:
sz = 768 #image size
bs = 2  #batch size

md = get_data(sz,bs)

learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam

learn.crit = MixedLoss(10.0, 2.0)
learn.metrics=[accuracy_thresh(0.5),dice,IoU]

learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)
#In [38]:
learn.fit(lrs/10,1,wds=wd,cycle_len=1,use_clr=(10,8))


learn.save('Unet34_768_1')

#==================================================================================
#==================================================================================
#==================================================================================

# step2
def cut_empty(names):
    return [name for name in names 
            if(type(segmentation_df.loc[name]['EncodedPixels']) != float)]

tr_n_cut = cut_empty(tr_n)
val_n_cut = cut_empty(val_n)

def get_score(pred, true):
    n_th = 10
    b = 4
    thresholds = [0.5 + 0.05*i for i in range(n_th)]
    n_masks = len(true)
    n_pred = len(pred)
    ious = []
    score = 0
    for mask in true:
        buf = []
        for p in pred: buf.append(IoU(p,mask))
        ious.append(buf)
    for t in thresholds:   
        tp, fp, fn = 0, 0, 0
        for i in range(n_masks):
            match = False
            for j in range(n_pred):
                if ious[i][j] > t: match = True
            if not match: fn += 1
        
        for j in range(n_pred):
            match = False
            for i in range(n_masks):
                if ious[i][j] > t: match = True
            if match: tp += 1
            else: fp += 1
        score += ((b+1)*tp)/((b+1)*tp + b*fn + fp)       
    return score/n_th

def split_mask(mask):
    threshold = 0.5
    threshold_obj = 30 #ignor predictions composed of "threshold_obj" pixels or less
    labled,n_objs = ndimage.label(mask > threshold)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if(obj.sum() > threshold_obj): result.append(obj)
    return result

def get_mask_ind(img_id, df, shape = (768,768)): #return mask for each ship
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return []
    if(type(masks) == str): masks = [masks]
    result = []
    for mask in masks:
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
        result.append(img.reshape(shape).T)
    return result

class Score_eval():
    def __init__(self):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        self.score, self.count = 0.0, 0
        
    def put(self,pred,name):
        true = get_mask_ind(name, self.segmentation_df)
        self.score += get_score(pred,true)
        self.count += 1
        
    def evaluate(self):
        return self.score/self.count

PRETRAINED_SEGMENTATION_PATH = PATH + 'models/'
learn = ConvLearner(md, models)
learn.models_path = PRETRAINED_SEGMENTATION_PATH
learn.load('Unet34_768_1')
learn.models_path = PATH

def model_pred(learner, dl, F_save): #if use train dl, disable shuffling
    learner.model.eval();
    name_list = dl.dataset.fnames
    num_batchs = len(dl)
    t = tqdm(iter(dl), leave=False, total=num_batchs)
    count = 0
    for x,y in t:
        py = to_np(F.sigmoid(learn.model(V(x))))
        batch_size = len(py)
        for i in range(batch_size):
            F_save(py[i],to_np(y[i]),name_list[count])
            count += 1
# Running the model evaluation on the validation set.
'''
score = Score_eval()
process_pred = lambda yp, y, name : score.put(split_mask(yp),name)
model_pred(learn, md.val_dl, process_pred)
print('\n',score.evaluate())
'''


md = get_data(sz,bs)
learn.set_data(md)
#The function for mask decoding is borrowed from https://www.kaggle.com/kmader/from-trained-u-net-to-submission-part-2/notebook .

#In [25]:
def decode_mask(mask, shape=(768, 768)):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
#In [26]:
ship_list_dict = []
#for name in test_names_nothing:
#    ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
#In [27]:
def enc_test(yp, y, name):
    masks = split_mask(yp)
    if(len(masks) == 0): 
        ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
    for mask in masks:
        ship_list_dict.append({'ImageId':name,'EncodedPixels':decode_mask(mask)})
#In [28]:
model_pred(learn, md.test_dl, enc_test)
pred_df = pd.DataFrame(ship_list_dict)
pred_df.to_csv(PATH+'submission.csv', index=False)


#pred_df = pd.read_csv(PATH+'submission.csv')





