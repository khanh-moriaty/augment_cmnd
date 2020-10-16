import numpy as np
import cv2

#******************************************************
# Purpose: Multiplies all values in a color channel of
# an image by a value, then cap them to [0, 255].
# This function is used to enhance green and blue
# channel in CMND images.
#
# Inputs:
#         img: numpy.ndarray object representing the image.
#         channel: the color channel to be multiplied.
#         value: the multiplying value.
#
# Returns: The original image after multiplying. 
# Notes: The original image is modified after
# this operation.
#
# Author: moriaty.
# Last modified: 2/4/2020.
#******************************************************

def multiply_channel(img, channel, value=1):
    tmp = img[:,:,channel].astype(np.float32)
    tmp = tmp*value
    tmp = np.clip(tmp, 0, 255)
    img[:,:,channel] = tmp.astype(np.uint8)
    return img

#******************************************************
# Purpose: Multiplies all values in a color channel of
# an image by a value, then cap them to [0, 255].
# This function is used to enhance green and blue
# channel in CMND images.
#
# Inputs:
#         img: numpy.ndarray object representing the image.
#         channel: the color channel to be multiplied.
#         value: the multiplying value.
#
# Returns: The original image after multiplying. 
# Notes: The original image is modified after
# this operation.
#
# Author: moriaty.
# Last modified: 9/4/2020.
#******************************************************

def add_channel(img, channel, value=1):
    tmp = img[:,:,channel].astype(np.float32)
    tmp = np.add(tmp,value)
    tmp = np.clip(tmp, 0, 255)
    img[:,:,channel] = tmp.astype(np.uint8)
    return img

#******************************************************
# Purpose: Transforms the perspective of an image to
# make it looks 3D-esque.
#
# Inputs:
#         img: numpy.ndarray object representing the
# image.
#         top: set to True if you want to make the top
# side looks further away from the screen. Otherwise
# this makes the bottom side looks further away.
#         left: set to True if you want to make the left
# side looks further away from the screen. Otherwise
# this makes the right side looks further away.
#         scale: how many percents of the side will be
# "pushed" further away.
#
# Returns: The image after transforming.
#
# Author: moriaty.
# Last modified: 2/4/2020.
#******************************************************
    
def perspective_transform(img, top=False, left=True, scale=.05):
    (rows, cols, dim) = img.shape

    p1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    # p2 = np.float32([[0, rows * scale], [cols, 0], [cols * scale, rows
                    # * (1 - scale)], [cols * (1 - scale), rows]])
    p2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    if top:
        p2[0,0] = cols * scale
        p2[1,0] = cols * (1-scale)
    else:
        p2[2,0] = cols * scale
        p2[3,0] = cols * (1-scale)
    if left:
        p2[0,1] = rows * scale
        p2[2,1] = rows * (1-scale)
    else:
        p2[1,1] = rows * scale
        p2[3,1] = rows * (1-scale)
    M = cv2.getPerspectiveTransform(p1, p2)
    M = np.array(M, dtype=np.float32)
    # print(str(M))
    img = cv2.warpPerspective(src=img, M=M, dsize=(cols, rows))
    return (img, M, p2)

#******************************************************
# Purpose: Creates a "dirty" folding line connecting
# two given points and insert it onto an image.
#
# Inputs:
#         img: cv2 image.
#         p1, p2: the two points connecting the line.
#         dmax: maximum distance of the trailing shadow.
#         l2r: set to 1 if you want the shadow to be drawn
# from left to right instead of from right to left.
#        max_alpha: maximum alpha value of the folding
# line and the shadow.
#        exp_base: base of the exponential function
# used to generate the shadow's alpha value.
#        exp_power: maximum power of the exponential
# function used to generate the shadow's alpha value.
#
# Returns: The image after creating and inserting
# a dirty folding line onto it.
#
# Author: moriaty.
# Last modified: 2/4/2020.
#******************************************************
    
def create_folding(img, p1, p2, dmax=80, l2r=0,
    max_alpha=64, exp_base=1.5, exp_power=10
    ):
    if p1[1] > p2[1]:
        p1, p2 = p2, p1
    
    height, width = img.shape[:2]
    px = np.zeros(shape=(height, width, 4), dtype=np.float32)
    
    dx = (p2[0] - p1[0]) / (p2[1] - p1[1])

    for i in range(dmax):
        alpha = max_alpha * (exp_base ** (exp_power * (1 - i / dmax)) - 1) / (exp_base ** exp_power - 1)
        alpha = int(alpha)
        for j in range(-dmax, height):
            x = int(p1[0] + dx * j + (2*l2r - 1) * i)
            y = int(p1[1] + j)
            if x >= 0 and x < width and y >= 0 and y <= p2[1] + dmax and y < height:
                px[y, x, 3] = alpha / 255

    img = px[:,:,3] * px[:,:,:3] + (1 - px[:,:,3]) * img
    img = img.astype(np.uint8)
    
    return img

#******************************************************
# Purpose: Blurs a portion of an image.
# This function is used to make the image looks like
# a dirty old man. ;)
#
# Inputs:
#         img: PIL image object representing the
# image.
#         box: array contains coordinates of top-left and
# bottom-right corners of the portion to be blurred.
#        kernel_radius: radius of the kernel used
# in Gaussian Blur.
#
# Returns: The image after creating and inserting
# a shiny bubble onto it.
# Notes: The original image is modified after this
# operation.
#
# Author: moriaty.
# Last modified: 3/4/2020.
#******************************************************

def blur(img, box, kernel_radius=1):
    blur = img.crop(tuple(box))
    blur = blur.filter(ImageFilter.GaussianBlur(kernel_radius))
    img.paste(blur, box)
    return img

#******************************************************
# Purpose: Creates a shiny illuminating bubble and
# insert it onto an image.
#
# Inputs:
#         img: cv2 image.
#         radius: the radius of the bubble to be created.
#        max_alpha: maximum opacity of the bubble.
#        exp_base: base of the exponential function
# used to generate the shadow's alpha value.
#        exp_power: maximum power of the exponential
# function used to generate the shadow's alpha value.
#        fill: a tuple representing RGB color of the
# bubble.
#
# Returns: The image after creating and inserting
# a shiny bubble onto it.
#
# Author: moriaty.
# Last modified: 3/4/2020.
#******************************************************

def create_bubble(
    img, pos,
    radius=100,
    max_alpha=200,
    exp_base=2.0,
    exp_power=5,
    fill=(255,255,255)
    ):
    
    height, width = img.shape[:2]
    px = np.zeros(shape=(height, width, 4), dtype=np.float32)
    
    x,y = pos
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            if x+i < 0 or x+i >= width:
                continue
            if y+j < 0 or y+j >= height:
                continue
            dist = (i**2 + j**2) ** .5
            if dist > radius:
                continue
            alpha = (exp_base ** (exp_power * (1 - dist / radius)) - 1) / (exp_base ** exp_power - 1)
            alpha = int(max_alpha * alpha) / 255
            px[y+j, x+i] = np.array(list(fill) + [alpha])
    
    img = px[:,:,3:] * px[:,:,:3] + (1 - px[:,:,3:]) * img
    img = img.astype(np.uint8)
    
    return img
