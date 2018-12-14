import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import cv2

def process_img(img_filename):
    X = mpimg.imread(img_filename)[:,:,0:3].reshape(-1,3)
    new_shape = list(mpimg.imread(img_filename).shape)
    new_shape[2] = 3
    X = X.reshape(new_shape)

    X_processed = tune_img(X)
    _, labels = cv2.connectedComponents(img)

    components = get_components(labels)
    return components




def pad(arr, pad_width):
    arr_new = np.hstack([np.zeros([arr.shape[0], pad_width]), arr]) #left
    arr_new = np.hstack([arr_new, np.zeros([arr_new.shape[0], pad_width])]) #right
    arr_new = np.vstack([np.zeros([pad_width, arr_new.shape[1]]), arr_new]) #up
    arr_new = np.vstack([arr_new, np.zeros([pad_width, arr_new.shape[1]])]) #down
    return arr_new

def square(arr, pad_width, top, left, bottom, right):
    arr_square = arr[top-pad_width:bottom+pad_width, left-pad_width:right+pad_width]
    diff = abs(arr_square.shape[1] - arr_square.shape[0])
    pad = diff // 2
    if arr_square.shape[0] < arr_square.shape[1]:
        arr_square = np.vstack([np.zeros([pad, arr_square.shape[1]]), arr_square]) #up
        arr_square = np.vstack([arr_square, np.zeros([pad + (diff % 2 == 1), arr_square.shape[1]])]) #down
    elif arr_square.shape[0] >= arr_square.shape[1]:
        arr_square = np.hstack([np.zeros([arr_square.shape[0], pad]), arr_square]) #left
        arr_square = np.hstack([arr_square, np.zeros([arr_square.shape[0], pad + (diff % 2 == 1)])]) #right
    return arr_square

def erode(arr, erosion_percent):
    for dim in range(1, 12):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dim, dim))
        erosion = cv2.erode(arr, kernel, iterations = 1)
        if np.sum(erosion)/np.sum(arr) < erosion_percent:
            break
    dim -= 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dim, dim))
    erosion = cv2.erode(arr, kernel, iterations = 1)
    return erosion

def get_components(labels, pad_width=3, erosion_percent=0.4):
    components = {i:{'label':None, 
                    'output':None, 
                    'tl':None, 
                    'br':None, 
                    'pic':None,
                    'group':None,
                    'sup':False,
                    'sub':False} 
                 for i in range(1, len(np.unique(labels)))}

    fig, axes = plt.subplots(2, int((len(components) + 1)/2), figsize=(15, 5))
    for i, ax in zip(sorted(components.keys()), axes.ravel()):
        label = labels.copy()
        label[labels != i] = 0
        label_padded = pad(label, pad_width)

        # Get dimensions of component
        xs, ys = np.where(label != 0)
        top, bottom, left, right = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
        components[i]['tl'] = (top, left)
        components[i]['br'] = (bottom, right)

        # Square and resize
        label_square = square(label_padded, pad_width, top, left, bottom, right)
        label_square = cv2.resize(label_square, (45, 45))
        label_square[label_square != 0] = 1

        # Erode based on size
        label_eroded = erode(label_square, erosion_percent)
        components[i]['pic'] = label_eroded.ravel()
    return components

def tune_img(X, dilation_kernel=None, dilation_iterations=3):
    # Binary
    X_gray = cv2.cvtColor(cv2.imread(file_name_X).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Remove initial noise and smoothen lighting gradient
    X_gray_smooth = cv2.GaussianBlur(X_gray, (11, 11), 0) #
    # Threshold
    X_gray_smooth = cv2.adaptiveThreshold(X_gray_smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Blur
    X_gray_smooth = cv2.blur(X_gray_smooth, (3, 3))
    # Threshold
    X_gray_smooth = cv2.threshold(X_gray_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Median blur
    X_gray_smooth = cv2.medianBlur(X_gray_smooth, 17)
    # Threshold
    X_bw = 1 - np.round(X_gray_smooth/255)

    # Dilate
    if not dilation_kernel: dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    X_bw =  cv2.dilate(X_bw, dilation_kernel, iterations=dilation_iterations)

    return X_bw.astype(np.uint8)