from PIL import Image
import math
import cv2
import numpy as np
from sklearn.cluster import KMeans

def read_image(img_path):
    """Read image when image path is given"""
    image = cv2.imread(img_path)

    return image

def get_colors(cluster, centroids, exact=False):
    """for a given image, get all the colors and their percentages"""
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    
    # Convert each RGB color code from float to int
    if not exact:
        centroids = centroids.astype("int")
    
    # get the colors of the image
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])

    return colors

def is_dark(image):
    """cheks whether the dominant color of the image is dark"""
    
    #converts the image to a list of pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))
    
    # Find and display most X dominant colors
    cluster = KMeans(n_clusters=5).fit(reshape)
    colors = get_colors(cluster, cluster.cluster_centers_)
    
    # Obtain dominant RGB color code
    dominant_color = colors[-1][1].tolist()
    dominant_color_average = int(sum(dominant_color) / 3)
    
    # dominant_color_average <= 85: -> Dark/Black
    # dominant_color_average > 85 and dominant_color_average <= 170 -> in between
    # dominant_color_average > 170: Light/White
    
    #return dominant color
    if dominant_color_average <= 85:
        return True
    else:
        return False

def inversion(img):
    """
    if an image has light text on dark background,
    inverts to get dark text on light background
    """
    if is_dark(img):
        return cv2.bitwise_not(img)
    else:
        return img

def super_res(img, ocr_model_path):
    """increase the image resolution using OpenCV's ESPCN deep learning model"""
    
    #Load the Lapsrn model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(ocr_model_path)
    sr.setModel("espcn", 3)

    #upsample and return the image
    return sr.upsample(img)

def grayscale(img): #binarization_1
    """grayscaling images"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blackwhite(gray_image): #binarization_2
    """making images black and white"""
    thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
    return im_bw
    
def noise_removal(image): #feed im_bw"
    """removes image noise"""
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thin_font(image):
    """makes bold fonts thinner - known as erosion"""
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def thick_font(image):
    """makes faint fonts bolder - known as dilation"""
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def remove_borders(image):
    """removes borders from images"""
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

def add_borders(image):
    """expands the edges, incase the letters start too close to the edge"""
    color = [255, 255, 255]
    top, bottom, left, right = [250]*4
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def main(img_path, ocr_model_path):
    """sequences the image preprocessing steps into a processing chain"""
    #read
    img = read_image(img_path)
    
    #invert
    inverted = inversion(img)

    #binarized
    binarized = blackwhite(inverted)

    #upscaled
    upscaled = super_res(binarized, ocr_model_path)

    #erosion
    eroded = thick_font(upscaled)

    #add_borders
    bordered = add_borders(eroded)

    return bordered