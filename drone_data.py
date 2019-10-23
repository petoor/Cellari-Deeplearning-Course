import wget
import os
import numpy as np
from PIL import Image

raw_image_url = 'http://www.lapix.ufsc.br/wp-content/uploads/2019/05/sugarcane2.png'
anno_image_url = 'http://www.lapix.ufsc.br/wp-content/uploads/2019/05/crop6GT.png'
crop_size = 512

raw_img_location = os.getcwd()
anno_img_location = os.getcwd()

raw_img_train_cropped_location = raw_img_location+os.sep+'train'+os.sep+'cropped'+os.sep+'raw'
raw_img_test_cropped_location = raw_img_location+os.sep+'test'+os.sep+'cropped'+os.sep+'raw'
anno_img_train_cropped_location = anno_img_location+os.sep+'train'+os.sep+'cropped'+os.sep+'anno'
anno_img_test_cropped_location = anno_img_location+os.sep+'test'+os.sep+'cropped'+os.sep+'anno'

if not os.path.isdir(raw_img_train_cropped_location):
    os.makedirs(raw_img_train_cropped_location)
if not os.path.isdir(raw_img_test_cropped_location):
    os.makedirs(raw_img_test_cropped_location)

if not os.path.isdir(anno_img_train_cropped_location):
    os.makedirs(anno_img_train_cropped_location)
if not os.path.isdir(anno_img_test_cropped_location):
    os.makedirs(anno_img_test_cropped_location)

wget.download(raw_image_url, raw_img_location+os.sep+'raw.png')
wget.download(anno_image_url, anno_img_location+os.sep+'anno.png')

raw_img = Image.open(raw_img_location+os.sep+'raw.png').convert("RGB")
anno_img = Image.open(anno_img_location+os.sep+'anno.png').convert("RGB")

size = (crop_size * int(np.ceil(raw_img.size[0] / crop_size)),
        crop_size * int(np.ceil(raw_img.size[1] / crop_size)))

raw_img_padded = Image.new("RGB", size, (0, 0, 0))
raw_img_padded.paste(raw_img, (0, 0))
raw_img_padded = np.array(raw_img_padded, dtype=np.uint8)

anno_img_padded = Image.new("RGB", size, (0, 0, 0))
anno_img_padded.paste(anno_img, (0, 0))
anno_img_padded = np.array(anno_img_padded, dtype=np.uint8)

np.random.seed(1337) #Seeding RNG.

x_range = raw_img_padded.shape[0] // crop_size
y_range = raw_img_padded.shape[1] // crop_size
rand_list = np.random.binomial(1, 0.25, x_range*y_range)
for x in range(x_range):
        for y in range(y_range):
                if rand_list[y+(x*y_range)] == 0:
                    cropped_img = raw_img_padded[crop_size*x:crop_size*(x+1), crop_size*y:crop_size*(y+1)]
                    Image.fromarray(cropped_img).save(raw_img_train_cropped_location+os.sep+str(x)+"_"+str(y)+"_"+'raw.png')
                    cropped_anno = anno_img_padded[crop_size*x:crop_size*(x + 1), crop_size*y:crop_size*(y + 1)]
                    Image.fromarray(cropped_anno).save(anno_img_train_cropped_location+os.sep+str(x) + "_"+str(y)+"_"+'anno.png')
                else:
                    cropped_img = raw_img_padded[crop_size*x:crop_size*(x+1), crop_size*y:crop_size*(y+1)]
                    Image.fromarray(cropped_img).save(raw_img_test_cropped_location+os.sep+str(x)+"_"+str(y)+"_"+'raw.png')
                    cropped_anno = anno_img_padded[crop_size*x:crop_size*(x + 1), crop_size*y:crop_size*(y + 1)]
                    Image.fromarray(cropped_anno).save(anno_img_test_cropped_location+os.sep+str(x) + "_"+str(y)+"_"+'anno.png')

#Gives us a 119/35 train-test split which is a 77 % / 23 % split.

#For Eskilds drone dataset, go to : https://onedrive.live.com/?authkey=%21AMgYO6epJGalP6A&id=9F7B46F5E539ED76%2164379&cid=9F7B46F5E539ED76
