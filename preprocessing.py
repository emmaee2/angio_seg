from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
import hashlib
import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pydmd import DMD
import SimpleITK as sitk

def load_video(filename):
    print("Loading", filename)
    cap = cv2.VideoCapture(filename)
    
    data = []
    frames = 0;

    success,image = cap.read()
    (width,height,depth) = image.shape

    while success: #frames <=120 :
        image = image.astype('float32') # convert to greyscale
        image = cv2.resize(image,(800,600))
#         data += [np.reshape(image, (image.shape[0]*image.shape[1],3))]
        data += [image[:,:,0]]
        frames += 1
        success,image = cap.read()
    #data = np.array(data[:,::2])
    data = np.array(data)
    #videos[filename] = data
    #print("got {} frames of {}".format(frames,filename))
    return data

# mean: 106.51858735889945
# std: 39.54606051306486
def process_video(filename, truth):
    p = load_video(filename)
    print(p.shape)
    p_truth = cv2.imread(truth)[:,:,0]//np.max(cv2.imread(truth)[:,:,0])
    p_data = np.reshape(p, (p.shape[0], 600,800))
    ims = p_data[(len(p_data)//2)-5:(len(p_data)//2)+15,:,:]
    return ims, p_truth

def process_video_dmd(filename, truth):
    p = load_video(filename)
    p_data = np.reshape(p, (p.shape[0], 600,800))
    DMD, back = get_DMD_feature_video(p_data)
    
    return DMD, back

def get_DMD_feature_video(data, dmd_rank = 5, start_frame = 1, frame_step = 1, frame_limit = 15):
    frames, width, height = np.shape(data)
    center =  np.copy(data[frames//2,:,:])
    
    dmd = DMD(svd_rank=dmd_rank)
    data_small = np.array([data[i].reshape(width*height) for i in range(start_frame,start_frame + min(frames,frame_limit),frame_step)]).T
    dmd.fit(data_small)

    th=0.01
    background_mode_idx =[i for i in range(dmd_rank) if (np.abs(dmd.eigs[i].real-1) < th and np.abs(dmd.eigs[i].imag) < th)]
    foreground_modes_idx = [i for i in range(dmd_rank) if i not in background_mode_idx]
    foreground_modes = dmd.modes[:,foreground_modes_idx]
    background_mode = dmd.modes[:,background_mode_idx]
    background =  background_mode.dot(dmd.dynamics[background_mode_idx,:])[:,-1]
    
    background = background.reshape((width,height))
    background_deep = np.array(background)
    DMD_features = center - (.8*np.absolute(background_deep))

    return DMD_features, np.absolute(background)


def get_view_hash(patient,view):
    """
    Ignores file prefixes.
    """
    pid = "Patient{}/{}".format(patient,view)
    hash_id = hashlib.sha1(bytearray(pid,"UTF-8")).hexdigest()
    return hash_id


def enhance_contrast(im_path):
    img_T1 = sitk.ReadImage(im_path)
    k_radius = 10

    # white tophat
    wth_filter = sitk.WhiteTopHatImageFilter()
    wth_filter.SetKernelRadius(k_radius)
    segw = wth_filter.Execute(img_T1)
    # black tophat
    bth_filter = sitk.BlackTopHatImageFilter()
    bth_filter.SetKernelRadius(k_radius)
    segb = bth_filter.Execute(img_T1)
    
    return sitk.GetArrayFromImage(img_T1 + segw - segb)


def load_data(path, output_vid):
    for i in range(len(path)):
        im = enhance_contrast(path[i])
        n = os.path.basename(path[i])
        cv2.imwrite(os.path.join(output_vid, n), im)
        print('Writing to ', os.path.join(output_vid, n))
    return   

data_dir = "./data/processed/"

centers = [data_dir + 'centerFrames/'+ i for i in os.listdir(data_dir + 'centerFrames')]
print('Images to be processed:', centers)
out_path = os.path.join(data_dir, "contrastEnhanced")
load_data(centers, out_path)