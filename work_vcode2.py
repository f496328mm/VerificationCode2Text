
# ref by https://github.com/f496328mm/simple-railway-captcha-solver

from PIL import Image, ImageDraw, ImageFont
#from random import randint
import pandas as pd
import numpy as np
import random
import os
import sys
path = os.listdir('/home')[0]
sys.path.append('/home/'+ path +'/github')
#os.chdir('./build_model/')
#os.getcwd()
FONTPATH = [ '/home/'+ path +'/github/VerificationCode2Text/' + te for te in ['Times Bold.ttf','Courier-BoldRegular.ttf'] ]

def image2EncodedPixels(captcha):
    captcha = np.array(captcha)
    captcha = captcha[:,:,0]
    captcha2 = captcha.T.reshape((60*200))
    
    value = []
    for i in range(len(captcha2)):
        if captcha2[i] != 0 :
            value.append(i)
           
    value2 = ''
    bo = 0
    total = 0
    for i in range(len(value)):
        #print( 'i = ' + str( i ))
        #print( 'bo = ' + str( bo ))
        if bo == 0:
            value2 = value2 + str(value[i]) + ' '
            bo = 1
        elif bo == 1: 
            if value[i] - value[i-1] == 1:
                total = total + 1
                
            elif value[i] - value[i-1] > 1:
                value2 = value2 + str(total) + ' '
                bo = 0
                total = 0
        if i == (len(value)-1) :
            value2 = value2 + str(total)   
            
    return value2

class rect:
    def __init__(self):
        self.size = (random.randint(5, 21), random.randint(5, 21))
        self.location = (random.randint(1, 199), random.randint(1, 59))
        self.luoverlay = True if random.randint(1, 10) > 6 else False
        self.rdoverlay = False if self.luoverlay else True if random.randint(1, 10) > 8 else False
        self.lucolor = 0 if random.randint(0, 1) else 255
        self.rdcolor = 0 if self.lucolor == 255 else 255
        self.ludrawn = False
        self.rddrawn = False
        self.pattern = random.randint(0, 1)

    def draw(self, image, overlay):
        if((overlay or not self.luoverlay) and not self.ludrawn):
            self.ludrawn = True
            stp = self.location
            transparent = int(255 * 0.45 if self.lucolor == 0 else 255 * 0.8)
            color = (self.lucolor, self.lucolor, self.lucolor, transparent)
            uline = Image.new("RGBA", (self.size[0], 1), color)
            lline = Image.new("RGBA", (1, self.size[1]), color)
            image.paste(uline, stp, uline)
            image.paste(lline, stp, lline)
        if((overlay or not self.rdoverlay) and not self.rddrawn):
            self.rddrawn = True
            dstp = (self.location[0], self.location[1] + self.size[1])
            rstp = (self.location[0] + self.size[0], self.location[1])
            transparent = int(255 * 0.45 if self.rdcolor == 0 else 255 * 0.8)
            color = (self.rdcolor, self.rdcolor, self.rdcolor, transparent)
            dline = Image.new("RGBA", (self.size[0], 1), color)
            rline = Image.new("RGBA", (1, self.size[1]), color)
            image.paste(dline, dstp, dline)
            image.paste(rline, rstp, rline)

A_Za_z = []
for i in range(65, 91):
    A_Za_z.append( chr(i) )
for i in range(10):
    A_Za_z.append(i)
    
# self = captchatext(1,0)
class captchatext:# priority = 1; offset = 0
    def __init__(self, priority, offset):
        
        self.number = random.sample(A_Za_z,1)[0]
        #self.number = randint(1,10)
        self.color = [random.randint(10, 140) for _ in range(3)]
        self.angle = random.randint(-55, 55)
        self.priority = priority
        self.offset = 0
        self.next_offset = 0
        #self.captcha = Image.new('RGBA', (200, 60))
    def draw(self, image,captcha):
        
        fontpath = FONTPATH[ random.sample(range(2),1)[0] ] 
        color = (self.color[0], self.color[1], self.color[2], 255)
        font = ImageFont.truetype( fontpath , random.randint(25, 27) * 10)
        text = Image.new("RGBA", (250, 300), (0, 0, 0, 0))
        textdraw = ImageDraw.Draw(text)
        
        textdraw.text((0, 0), str(self.number), font=font, fill=color)
        #textdraw.text((0, 0), 'j', font=font, fill=color)

        text = text.rotate(self.angle, expand=True)
        text = text.resize((int(text.size[0] / 10), int(text.size[1] / 10)))
        base = int(self.priority * (200 / 6))
        rand_min = (self.offset - base - 2) if (self.offset - base - 2) >= -15 else -15
        rand_min = 0 if self.priority == 0 else rand_min
        rand_max = (33 - text.size[0]) if self.priority == 5 else (33 - text.size[0] + 10)
        try:
            displace = random.randint(rand_min, rand_max)
        except:
            displace = rand_max
            
        location = (base + displace, random.randint(3, 23))
        self.next_offset = location[0] + text.size[0]
        image.paste(text, location, text)

        rectangle = Image.new("RGBA", (35, 35), (255,255,255,255))
        captcha.paste(rectangle, location, rectangle)

'''
data = pd.DataFrame()

value = work_vcode_fun('train5',5)
data = data.append(value)

'''
def work_vcode_fun(file_path,amount2):# amount = 5 ; file_path = 'train_data5'
    
    #os.chdir('/home/linsam/project/fb_chatbot/verification_code2text')
    ls = '/home/'+ path +'/github/VerificationCode2Text/'
    if file_path in os.listdir(ls):
        print(1)
    else:
        os.makedirs( ls + file_path)
        
    #numberlist = []
    #status = 1
    #print(index)
    # index = 0
    numberstr = ""
    bgcolor = [random.randint(180, 250) for _ in range(3)]
    image = Image.new('RGBA', (200, 60), (bgcolor[0], bgcolor[1], bgcolor[2], 255))
    rectlist = [rect() for _ in range(32)]
    for obj in rectlist:
        obj.draw(image=image, overlay=False)

    offset = 0
    #vcode = ''
    #amount2 = random.sample([5,6],1)[0]
    data = pd.DataFrame()
    EncodedPixels = []
    
    for i in range(amount2):# i = 0
        captcha = Image.new('RGBA', (200, 60),(0,0,0,255))
        newtext = captchatext(i, offset)
        newtext.draw(image = image, captcha = captcha)
        offset = newtext.next_offset
        numberstr += str(newtext.number)
        
        value = image2EncodedPixels(captcha)
        EncodedPixels.append( value )

    #captcha
    tem = '/home/'+ path +'/github/VerificationCode2Text/'
    data['EncodedPixels'] = EncodedPixels
    data['ImageId'] = numberstr + ".jpg"
    
    image.convert("RGB").save(tem + file_path + "/" + numberstr + ".jpg", "JPEG")
    return data

'''
====================
all_masks = np.zeros((60, 200))
for mask in EncodedPixels:
    #print(1)
    all_masks += rle_decode(mask,shape = (200,60))
    
fig, axarr = plt.subplots(1, 1, figsize=(15, 40))
axarr.imshow(np.array(image))
axarr.imshow(np.array(all_masks) , alpha=0.4)
#axarr[0].axis('off')
#axarr[1].axis('off')
#axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()
'''
