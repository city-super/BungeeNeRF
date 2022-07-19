import numpy as np
import os
import json
import cv2
import imageio

def _load_google_data(basedir, factor=None):
    img_basedir = basedir
    img_folder = 'images'
    imgdir = os.path.join(img_basedir, img_folder)
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')]
    sh = np.array(cv2.imread(imgfiles[0]).shape)
    imgs = []
    for f in imgfiles:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if im.shape[-1] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        im = cv2.resize(im, (sh[1]//factor, sh[0]//factor), interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32) / 255
        imgs.append(im)
    imgs = np.stack(imgs, -1) 
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)

    data = json.load(open(os.path.join(basedir, 'poses_enu.json')))
    poses = np.array(data['poses'])[:, :-2].reshape([-1, 3, 5])
    poses[:, :2, 4] = np.array(sh[:2]//factor).reshape([1, 2])
    poses[:, 2, 4] = poses[:,2, 4] * 1./factor 

    scene_scaling_factor = data['scene_scale']
    scene_origin = np.array(data['scene_origin'])
    scale_split = data['scale_split']

    return imgs, poses, scene_scaling_factor, scene_origin, scale_split

def load_multiscale_data(basedir, factor=3):
    imgs, poses, scene_scaling_factor, scene_origin, scale_split = _load_google_data(basedir, factor=factor)
    print('Loaded image data shape:', imgs.shape, ' hwf:', poses[0,:,-1])
    return imgs, poses, scene_scaling_factor, scene_origin, scale_split
    
    