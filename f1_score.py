from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
import glob
import os


class ColorChange:
    def __init__(self):
        self.colors = {
            'background': [0, 0, 0, 0],
            'black': [0, 0, 0, 255],  # Cell - Gland
            'green': [0, 255, 0, 255],  # Crop - Brazil
            'yellow': [255, 255, 0, 255],  # Weed - Brazil
            'gray': [128, 128, 128, 255], # Weed - Eskild
            'white': [255, 255, 255, 255], # Crop - Eskild
        }

    def rgba_to_mask(self, img):
        ''' Input (H,W,4), output (H,W) '''
        mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        for i, color_value in enumerate(list(self.colors.values())):
            mask[np.sum(img == color_value, axis=2) == 4] = i
        return mask

project = 'gland_images'
dataset = 'gland'

annotations_path = os.getcwd()+os.sep+project+os.sep+dataset+'/test/annotations'
predictions_path = os.getcwd()+os.sep+project+os.sep+dataset+'/test/predictions'
score_path = os.getcwd()+os.sep+project+os.sep+dataset

annotations_list = sorted(glob.glob(annotations_path+'/*.png'))
predictions_list = sorted(glob.glob(predictions_path+'/*.png'))
f1 = 0

for idx in range(len(predictions_list)):
    anno = Image.open(annotations_list[idx]).convert('RGBA')
    anno = np.uint8(anno)
    anno = ColorChange().rgba_to_mask(anno)
    pred = Image.open(predictions_list[idx]).convert('RGBA')
    pred = np.uint8(pred)
    pred = ColorChange().rgba_to_mask(pred)
    for classes in np.unique(anno):
        if classes == 0:
            continue
    y_pred = (pred==classes)*1
    y_true = (anno==classes)*1
    f1 += f1_score(y_true.reshape(-1), y_pred.reshape(-1))

f1 /= len(predictions_list)
f = open(score_path+"/f1_score.txt", "w")
f.write("The f1 score is : {0}".format(f1))
f.close()

print("F1 score calculated for {0}".format(dataset))
