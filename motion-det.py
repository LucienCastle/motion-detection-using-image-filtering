# -*- coding: utf-8 -*-

import os
import cv2
from google.colab.patches import cv2_imshow
from scipy.ndimage import correlate
import numpy as np

def getImageFilenames(root_dir):
  imgs_names = os.listdir(root_dir)
  imgs_names.sort()

  im = cv2.imread(os.path.join(root_dir,imgs_names[0]))
  height, width, _ = im.shape

  return imgs_names, height, width

def captureFrames(
    root_dir:str, # path to the root image folder
    img_paths:list, # list of image paths 
    sp_filter:np.ndarray # spatial filter for smoothing
)->np.ndarray:
  '''
  Captures multiple frames and applying smoothing filter
  '''
  frames = []
  for img in img_paths:
    # read images one by one
    im = cv2.imread(os.path.join(root_dir ,img))

    # convert image to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # apply smoothing
    smoothed = cv2.filter2D(im, -1, sp_filter)
    frames.append(smoothed)
  return np.stack(frames, axis=0)

def applyTemporalFilter(
    frames:np.ndarray, # smoothed frames 
    t_filter:np.ndarray, # temporal filter
)->np.ndarray:
  '''
  applies temporal filter
  '''
  filtered_img = np.apply_along_axis(correlate, axis=0, arr=frames, weights=t_filter)
  return filtered_img

def applyThresholding(
    frames:np.ndarray, # frames with after applying temporal filter 
    og_frames:np.ndarray, # original frames
    thresh:int=1.5, # thresholding values
)->np.ndarray:
  '''
  Applies thresholding
  '''
  thresh_res = []

  for i in range(frames.shape[0]):
    # create thresholding mask
    motion_mask = cv2.threshold(frames[i], np.std(frames[i]), 1, cv2.THRESH_BINARY)[1]

    # add mask to the original frame
    res = cv2.bitwise_and(og_frames[i], og_frames[i], mask=motion_mask)
    thresh_res.append(res)
  return np.stack(thresh_res, axis=0)

def gaussian_derivative_1d(
    tsigma:int, # standard deviation 
    size:int, # size of the filter
)->np.ndarray:
    """
    Generate a 1D derivative of Gaussian kernel with standard deviation tsigma
    """
    x = np.linspace(-(size-1) / 2., (size-1) / 2., size)
    kernel = -x / (tsigma ** 2) * np.exp(-0.5 * np.square(x) / np.square(tsigma))
    return kernel

if __name__ == '__main__':
  image_dir_path = input("Enter path of the images folder:")

  # retrieve image filenames
  img_fnames, height, width = getImageFilenames(image_dir_path)

  temporal_filter_type = 'gaussian'  # Options: 'simple' or 'gaussian'
  temporal_size = 3
  temporal_sigma = 1.4

  spatial_filter_type = 'gaussian'  # Options: 'box', 'gaussian'
  spatial_size = 5 # Options: 3, 5
  spatial_sigma = 1.4

  thresh = 1.5  # Threshold factor for determining motion mask

  # Define filters
  if temporal_filter_type == 'simple':
    temporal_filter = np.array([-0.5, 0, 0.5])
  else:
    temporal_filter = gaussian_derivative_1d(temporal_sigma, temporal_size)

  if spatial_filter_type == 'box':
    spatial_filter = np.ones((spatial_size, spatial_size), np.float32) / (spatial_size**2)
  else:
    spatial_filter = cv2.getGaussianKernel(spatial_size, spatial_sigma)
    spatial_filter = np.outer(spatial_filter, spatial_filter)

  i = 0 # options: 287 for office, 27 for red chair, start from 0 to generate the entire video
  video_name = 'video_red.avi'

  # define the video writer
  fourcc = cv2.VideoWriter_fourcc(*"MJPG")
  video = cv2.VideoWriter(video_name, fourcc, 30., (width, height), 0)

  while i < len(img_fnames):
    # capture frames and apply smoothing
    cap = captureFrames(image_dir_path, img_fnames[i:i+temporal_size], spatial_filter)

    # apply temporal filter
    temp_filtered = applyTemporalFilter(cap, temporal_filter)

    # apply thresholding
    thresh_img = applyThresholding(temp_filtered, cap, thresh)
    
    # uncomment the following 2 lines to generate the video
    for im in thresh_img:
      video.write(im)
    
    # for i in range(thresh_img.shape[0]):
    #   cv2_imshow(thresh_img[i])

    i += temporal_size

    # comment the break statement and for loop of cv2_imshow statements to avoid getting output for each frames
    # break

  # uncomment the following 2 lines while generating video
  cv2.destroyAllWindows()
  video.release()

R