import sys
import os
import glob

import torch
import numpy as np
import cv2
import tqdm
import pandas as pd
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

sys.path.append('.')
from scripts.detector import Detector


ID_TO_NAME = ['negative ', 'typical ', 'indeterminate ', 'atypical ']
D_BOX = ' 0 0 1 1'

def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    if fix_monochrome and dicom.PhotometricInterpretation == 'MONOCHROME1':
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    im = Image.fromarray(array)
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return im

def detect(model, img_path, img_size):
    img = read_xray(img_path)
    scale_x = img.shape[1] / img_size
    scale_y = img.shape[0] / img_size
    
    im = resize(img, img_size)
    bboxes, scores, probs = model.detect_image(im)
    if bboxes.shape[0]:
        bboxes[:, 0] = bboxes[:, 0] * scale_x
        bboxes[:, 2] = bboxes[:, 2] * scale_x
        bboxes[:, 1] = bboxes[:, 1] * scale_y
        bboxes[:, 3] = bboxes[:, 3] * scale_y
    return bboxes, scores, probs

def create_img_string(bboxes, scores):
    if len(bboxes) == 0:
        pred_str = 'none 1 0 0 1 1'
    else:
        pred_str = ''
        for bbox, score in zip(bboxes, scores):
            if pred_str != '':
                pred_str += ' '
            # opacity 0.5 100 100 200 200
            pred_str += f'opacity {score} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}'
    return pred_str
        

if __name__ == '__main__':
    IMG_SIZE = 640
    pt_path = '/data/aza_s/SIIM/flexible-yolov5/runs/train/exp2/weights/best.pt'
    model = Detector(pt_path, None, IMG_SIZE, xcycwh=False)
    imgs_root = '/data/zhan/compets/siim_covid/input/test_dcm'
    sub_df_path = '/data/zhan/compets/siim_covid/input/sample_submission.csv'
    imgs = os.listdir(imgs_root)
    save_dir = '/data/aza_s/SIIM/dataset/out'
    df = pd.read_csv(sub_df_path)
    study_to_img = {}
    img_to_path = {}

    for i, row in df.iterrows():
        name = row['id']
        if 'study' in name:
            name = name.split('_')[0]
            study_dir = os.path.join(imgs_root, name)
            imgs = glob.glob(os.path.join(study_dir, '*/*dcm'))
            assert len(imgs) > 0, name
            imgs = [img.split('/')[-1][:-4] for img in imgs]
            study_to_img[name] = imgs

    imgs = glob.glob(os.path.join(imgs_root, '*/*/*dcm'))
    for img in imgs:
        assert len(img.split('/')) > 2, img
        name = img.split('/')[-1][:-4]
        img_to_path[name] = img

    preds = {}
    new_imgs, strings = [], []
    for i, row in tqdm.tqdm(df.iterrows()):
        name = row['id']
        new_imgs.append(name)
        if 'study' in name:
            imgs = study_to_img[name.split('_')[0]]
            assemble = []
            for img_hash in imgs:
                img_path = img_to_path[img_hash]
                bboxes, scores, probs = detect(model, img_path, IMG_SIZE)
                img_level = img_path.split('/')[-1][:-4] + '_image'
                preds[img_level] = create_img_string(bboxes, scores)
                assemble.append(probs)
            probs = np.concatenate(assemble, 0).max(axis=0)
            study_str = ''
            for i in range(4):
                if study_str != '':
                    study_str += ' '
                study_str += ID_TO_NAME[i] + f'{probs[i]}' + D_BOX
            strings.append(study_str)
        else:
            if name not in preds:
                try:
                    img_path = img_to_path[name.split('_')[0]]
                    bboxes, scores, probs = detect(model, img_path, IMG_SIZE)
                    img_str = create_img_string(bboxes, scores)
                except Exception as e:
                    print(e)
            else:
                img_str = preds[name]
            strings.append(img_str)
    pred_df = pd.DataFrame({'id': new_imgs, 'PredictionString': strings})
    pred_df.to_csv("out.csv", index=False)
