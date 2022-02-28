import numpy as np
import cv2
import mahotas
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

def haralick(image):   
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = mahotas.features.haralick(grey).mean(axis=0)
    return feature

def zernikeMoments(image, radius=21, degree=8):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = mahotas.features.zernike_moments(grey, radius, degree)
    return feature

def colorHisto(image, n_bins=8):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    histo  = cv2.calcHist([hsv], [0, 1, 2], None, [n_bins, n_bins, n_bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(histo, histo)
    feature = histo.flatten()
    return feature

def lbp(image, numPoints=24, radius=8):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(grey, numPoints, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    feature, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return feature

def h_u(image):    
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(grey)).flatten()
    return feature

def extract_features(image):    
    h_u_moment = h_u(image)
    zernik = zernikeMoments(image)
    haralick_   = haralick(image)
    lbp_hist  = lbp(image)
    color_hist  = colorHisto(image)
    feature = np.hstack([h_u_moment, zernik, haralick_, lbp_hist, color_hist])
    return feature
