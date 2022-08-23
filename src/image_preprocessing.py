from PIL import Image
import math

# import ocr
# import ocr evaluation steps
# https://www.youtube.com/watch?v=ADV-AjAXHdc&t=1698s
# 
        
def isLightOrDark(rgbColor):
    """checks whether a given pixel is light or dark"""
    [r,g,b] = rgbColor
    hsp = math.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    if (hsp>127.5):
        return True
    else:
        return False
    
def binarization(image_path):
    """converts a color image to black and white"""
    image = Image.open(image_path)
    im = image.convert("RGB")
    pixelMap = im.load()
    
    img = Image.new(im.mode, im.size)
    pixelsNew = img.load()
    
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            #if light
            if isLightOrDark(list(pixelMap[i,j])): #function which checks light or dark
                pixelMap[i,j] = (0,0,0)
            #if dark
            else:
                pixelsNew[i,j] = (254,254,254)
    return img


   
    
    
    
  
    
    
   
    
    