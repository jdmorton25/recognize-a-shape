import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib as mpl
from random import randint, uniform
from tqdm import tqdm
import os

train_samples, test_samples = 3000, 1000

import cv2

pic_size = 64

def create_dataset(path, count):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        # create_circle(path, count)
        create_ellipse(path, i)
        # create_square(path, count)
        create_rectangle(path, i)
        create_triangle(path, i)

def rotate_image(image, center, theta, width, height):
    # https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
    theta *= np.pi / 180 # convert to rad

    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

def create_circle(path='./', number=0):
    image = np.full((pic_size, pic_size, 3), 255.)

    thickness = randint(1, 8)
    radius = randint(12, int(pic_size/3))
    center = (randint(radius + thickness, pic_size - thickness), 
              randint(radius + thickness, pic_size - thickness))
    color = (randint(0, 255), randint(0, 255), randint(0, 255))

    while not (thickness + radius < center[0] < pic_size - (thickness + radius) and thickness + radius < center[1] < pic_size - (thickness + radius)):
        center = (randint(radius + thickness, pic_size - thickness), 
                  randint(radius + thickness, pic_size - thickness))

    cv2.circle(img=image, center=center, radius=radius, color=color, thickness=thickness)

    cv2.imwrite('{}circle{:04}.jpg'.format(path, number), image)

def create_ellipse(path='./', number=0):
    image = np.full((pic_size, pic_size, 3), 255.)

    thickness = randint(2, 5)
    axes = (randint(4, int(pic_size/3)), randint(4, int(pic_size/3)))
    maxAx = max(axes[0], axes[1])
    center = (randint(maxAx + thickness, pic_size - thickness), 
              randint(maxAx + thickness, pic_size - thickness))
    color = (randint(0, 255), randint(0, 255), randint(0, 255))
    angle = randint(0, 360)

    while not (thickness + maxAx < center[0] < pic_size - (thickness + maxAx) and
               thickness + maxAx < center[1] < pic_size - (thickness + maxAx)):
        center = (randint(maxAx + thickness, pic_size - thickness), 
                  randint(maxAx + thickness, pic_size - thickness))

    cv2.ellipse(img=image, center=center, axes=axes, 
                angle=angle, startAngle=0, endAngle=360, 
                color=color, thickness=thickness)
    
    cv2.imwrite('{}ellipse{:04}.jpg'.format(path, number), image)

def create_square(path='./', number=0):
    image = np.full((pic_size, pic_size, 3), 255.)
    
    thickness = randint(2, 5)
    size = randint(10, pic_size/2)
    color = (randint(0, 255), randint(0, 255), randint(0, 255))
    angle = randint(0, 360)
    center = (randint(int(size/2 * np.sqrt(2) + thickness), pic_size - (int(thickness + size/2 * np.sqrt(2)))), 
              randint(int(size/2 * np.sqrt(2) + thickness), pic_size - (int(thickness + size/2 * np.sqrt(2)))))
    
    while not (size//2 + thickness < center[0] < pic_size - (size//2 + thickness) and
               size//2 + thickness < center[1] < pic_size - (size//2 + thickness)):
        center = (randint(int(size/2 * np.sqrt(2) + thickness), pic_size - (int(thickness + size/2 * np.sqrt(2)))), 
                  randint(int(size/2 * np.sqrt(2) + thickness), pic_size - (int(thickness + size/2 * np.sqrt(2)))))
    
    cv2.rectangle(img=image, pt1=(center[0] - size//2, center[1] - size//2),
                             pt2=(center[0] + size//2, center[1] + size//2), 
                             color=color, thickness=thickness)
    
    image = rotate_image(image, (int(pic_size/2), int(pic_size/2)), angle, pic_size, pic_size)
    
    cv2.imwrite('{}square{:04}.jpg'.format(path, number), image)

def create_rectangle(path='./', number=0):
    image = np.full((pic_size, pic_size, 3), 255.)
    
    thickness = randint(2, 5)
    size = (randint(10, pic_size//4), randint(9, pic_size/2))
    color = (randint(0, 255), randint(0, 255), randint(0, 255))
    angle = randint(0, 360)
    center = (randint(int(size[0]/2 * np.sqrt(2) + thickness), pic_size - (int(thickness + size[0]/2 * np.sqrt(2)))), 
              randint(int(size[1]/2 * np.sqrt(2) + thickness), pic_size - (int(thickness + size[1]/2 * np.sqrt(2)))))
    
    while not (size[0]//2 + thickness < center[0] < pic_size - (size[0]//2 + thickness) and
               size[1]//2 + thickness < center[1] < pic_size - (size[1]//2 + thickness)):
        center = (randint(int(size[0]/2 * np.sqrt(2) + thickness), pic_size - (int(thickness + size[0]/2 * np.sqrt(2)))), 
                  randint(int(size[1]/2 * np.sqrt(2) + thickness), pic_size - (int(thickness + size[1]/2 * np.sqrt(2)))))
    
    cv2.rectangle(img=image, pt1=(center[0] - size[0]//2, center[1] - size[1]//2),
                             pt2=(center[0] + size[1]//2, center[1] + size[1]//2), 
                             color=color, thickness=thickness)
    
    image = rotate_image(image, (int(pic_size/2), int(pic_size/2)), angle, pic_size, pic_size)
    
    cv2.imwrite('{}rectangle{:04}.jpg'.format(path, number), image)

def create_triangle(path='./', number=0):
    image = np.full((pic_size, pic_size, 3), 255.)
    
    thickness = randint(2, 5)
    scale = randint(2, 20)
    randcoef = randint(20, 50)
    color = (randint(0, 255), randint(0, 255), randint(0, 255))
    angle = randint(0, 360)
    center = (randint(int(pic_size//2 - 1/np.sqrt(3) * scale), int(pic_size//2 + 1/np.sqrt(3) * scale)),
              randint(int(pic_size//2 - 1/np.sqrt(3) * scale), int(pic_size//2 + 1/np.sqrt(3) * scale)))
    
    pts = np.array([[-1/2, -1/(2*np.sqrt(3))], [0, 1/np.sqrt(3)], [1/2, -1/(2*np.sqrt(3))]]) * randcoef
    pts = center + pts
    pts = np.array(pts.tolist(), np.int32)
    pts = pts.reshape((-1,1,2))
    
    cv2.polylines(img=image, pts=[pts], isClosed=True, color=color, thickness=thickness)
    
    image = rotate_image(image, (pic_size//2, pic_size//2), angle, pic_size, pic_size)
    
    cv2.imwrite('{}triangle{:04}.jpg'.format(path, number), image)

create_dataset('./data/train/', train_samples)
create_dataset('./data/test/', test_samples)