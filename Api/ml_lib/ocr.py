import numpy as np
import matplotlib.image as mpimg
import cv2

def process_img(img_filename):
    X = mpimg.imread(img_filename)[:,:,0:3].reshape(-1,3)
    new_shape = list(mpimg.imread(img_filename).shape)
    new_shape[2] = 3
    X = X.reshape(new_shape)

    X_gray = cv2.cvtColor(cv2.imread(img_filename).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    X_gray = X_gray.astype(np.uint8)
    X_gray_smooth = cv2.GaussianBlur(X_gray, (9, 9), 0) # remove initial noise and smoothen lighting gradient
    X_gray_smooth = cv2.adaptiveThreshold(X_gray_smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    X_gray_smooth = cv2.blur(X_gray_smooth, (3, 3))
    X_gray_smooth = cv2.threshold(X_gray_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    X_gray_smooth = cv2.medianBlur(X_gray_smooth, 13)
    X_bw = 1 - np.round(X_gray_smooth/255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    X_bw =  cv2.dilate(X_bw, kernel, iterations = 3)

    img = X_bw.astype(np.uint8)
    _, labels = cv2.connectedComponents(img)

    all_labels = []
    all_locations = []
    label_to_location = {}
    pad_width = 3
    
    for i in range(1, len(np.unique(labels))):
        l = labels.copy()
        l[labels != i] = 0
        
        # Pad
        l = np.hstack([np.zeros([l.shape[0], pad_width]), l]) #left
        l = np.hstack([l, np.zeros([l.shape[0], pad_width])]) #right
        l = np.vstack([np.zeros([pad_width, l.shape[1]]), l]) #up
        l = np.vstack([l, np.zeros([pad_width, l.shape[1]])]) #down
        
        # Get dimensions of component
        xs, ys = np.where(l != 0)
        top = np.min(xs)
        left = np.min(ys)
        bottom = np.max(xs)
        right = np.max(ys)
    
        # Square and resize
        l_padded = l[top-pad_width:bottom+pad_width, left-pad_width:right+pad_width]
        diff = abs(l_padded.shape[1] - l_padded.shape[0])
        pad = diff // 2
        if l_padded.shape[0] < l_padded.shape[1]:
            l_square = np.vstack([np.zeros([pad, l_padded.shape[1]]), l_padded]) #up
            l_square = np.vstack([l_square, np.zeros([pad + (diff % 2 == 1), l_padded.shape[1]])]) #down
        elif l_padded.shape[0] >= l_padded.shape[1]:
            l_square = np.hstack([np.zeros([l_padded.shape[0], pad]), l_padded]) #left
            l_square = np.hstack([l_square, np.zeros([l_padded.shape[0], pad + (diff % 2 == 1)])]) #right
        l_resized = cv2.resize(l_square, (45, 45))
        l_resized[l_resized != 0] = 1
        
        # Erode based on size
        all_locations.append(((top, left), (bottom, right)))
        label_to_location[i] = {'tl':(top, left), 'br':(bottom, right)}
        for dim in range(1, 12):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dim, dim))
            erosion = cv2.erode(l_resized, kernel, iterations = 1)
            if np.sum(erosion)/np.sum(l_resized) < 0.4:
                break
        dim -= 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dim, dim))
        erosion = cv2.erode(l_resized, kernel, iterations = 1)
        all_labels.append(erosion)
    return {'labels': all_labels, 'locations': label_to_location}