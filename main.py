import os
import cv2
import random
import math
import time
import numpy as np
import image_processing as img_proc
import xml.etree.ElementTree as ET
import xml.dom.minidom
from multiprocessing import Pool

INP_DIR = '/dataset/crawl/front_cmtnd_resized/'
OUT_DIR = '/dataset/augmented/front_cmtnd_resized/'
BGS_DIR = '/dataset/bg/'

def generate_bubble(img,
    radius=100,
    max_alpha=255,
    exp_base=1.5,
    exp_power=5,
    fill=(255,255,255)
    ):
    height, width = img.shape[:2]
    x = random.randint(width//6, width*5//6)
    y = random.randint(height//6, height*5//6)
    radius = height // 2
    return img_proc.create_bubble(img, (x,y), radius, max_alpha, exp_base, exp_power, fill)

def load_label(inp_label_path):
    tree = ET.parse(inp_label_path)
    label = tree.getroot()
    return label

def load_bbox(img, label):
    bbox = {}
    for obj in label.iter('object'):
        name = obj.findtext('name')
        bndbox = obj.find('bndbox')
        xmin = bndbox.findtext('xmin')
        xmax = bndbox.findtext('xmax')
        ymin = bndbox.findtext('ymin')
        ymax = bndbox.findtext('ymax')
        bndbox = [[xmin, ymin],
                  [xmax, ymin],
                  [xmax, ymax], 
                  [xmin, ymax]]
        bndbox = [[int(x) for x in y] for y in bndbox]
        bbox[name] = bndbox
        
        
    height, width = img.shape[:2]
    bbox_size = 10
    if 'topleft' not in bbox:
        bbox['topleft'] = [[1,1], [bbox_size,1], [bbox_size,bbox_size], [1,bbox_size]]
    if 'topright' not in bbox:
        bbox['topright'] = [[width-bbox_size,1], [width,1], [width,bbox_size], [width-bbox_size,bbox_size]]
    if 'botleft' not in bbox:
        bbox['botleft'] = [[1,height-bbox_size], [bbox_size,height-bbox_size], [bbox_size,height], [1,height]]
    if 'botright' not in bbox:
        bbox['botright'] = [[width-bbox_size,height-bbox_size], [width,height-bbox_size], [width,height], [width-bbox_size,height]]
        
    return bbox

def paste_onto_bg(img, bbox):
    bgs_dir = os.listdir(BGS_DIR)
    bg_path = os.path.join(BGS_DIR, random.sample(bgs_dir, 1)[0])
    bg = cv2.imread(bg_path)
    
    height, width = img.shape[:2]
    max_size = max(img.shape[:2])
    max_size_rotate = int(max_size * 1.41)
    bg_height, bg_width = [int((random.random() + 2) * max_size_rotate / 2) for x in bg.shape[:2]]
    bg = cv2.resize(bg, (bg_width, bg_height))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    random_angle = random.randint(1, 5) if random.random() < 0.5 else 0
    
    x_offset, y_offset = (max_size_rotate - width) // 2, (max_size_rotate - height) // 2
    aff_mat = np.array([[1, 0, x_offset], [0, 1, y_offset]], dtype=np.float32)
    img = cv2.warpAffine(img, aff_mat, (max_size_rotate, max_size_rotate), flags=cv2.INTER_LINEAR)
    bbox = {name:[np.dot(aff_mat, [x[0], x[1], 1]).tolist() for x in bb] for name, bb in bbox.items()}
    
    aff_mat = cv2.getRotationMatrix2D((max_size_rotate // 2, max_size_rotate // 2), random_angle, 1.0)
    img = cv2.warpAffine(img, aff_mat, (max_size_rotate, max_size_rotate), flags=cv2.INTER_LINEAR)
    bbox = {name:[np.dot(aff_mat, [x[0], x[1], 1]).tolist() for x in bb] for name, bb in bbox.items()}
    # img[:,:,3] = 255
    
    x_offset, y_offset = bg_width - max_size_rotate, bg_height - max_size_rotate
    x_offset, y_offset = [random.randint(0, int(offset))  for offset in [x_offset, y_offset]]
    
    img = img.astype(np.float)
    bg = bg.astype(np.float)
    paste_region = bg[y_offset:y_offset+max_size_rotate, x_offset:x_offset+max_size_rotate]
    bg[y_offset:y_offset+max_size_rotate, x_offset:x_offset+max_size_rotate] = (img[:,:,:3] * img[:,:,3:] + paste_region * (255 - img[:,:,3:])) / 255
    bg = bg.astype(np.uint8)
    
    scale = random.uniform(0.4, 1)
    bg = cv2.resize(bg, (int(bg_width * scale), int(bg_height * scale)))
    
    bbox = {name:[[(x[0] + x_offset) * scale, (x[1] + y_offset) * scale] for x in bb] for name, bb in bbox.items()}
    
    return bg, bbox
        
def save_label(img, label, bbox, out_label_path):
    label = ET.Element('annotation')
    sub = ET.SubElement(label, 'folder')
    sub.text = "image"
    sub = ET.SubElement(label, 'filename')
    sub.text = "404 ERROR"
    sub = ET.SubElement(label, 'path')
    sub.text = "404 ERROR"
    sub = ET.SubElement(label, 'source')
    sub = ET.SubElement(sub, 'database')
    sub.text = "Unknown"
    sub = ET.SubElement(label, 'segmented')
    sub.text = "0"
    
    height, width = img.shape[:2]
    sub = ET.SubElement(label, 'size')
    size = ET.SubElement(sub, 'width')
    size.text = str(width)
    size = ET.SubElement(sub, 'height')
    size.text = str(height)
    size = ET.SubElement(sub, 'depth')
    size.text = "3"
    
    for name, bb in bbox.items():
        obj = ET.SubElement(label, 'object')
        ele = ET.SubElement(obj, 'name')
        ele.text = name
        ele = ET.SubElement(obj, 'pose')
        ele.text = "Unspecified"
        ele = ET.SubElement(obj, 'truncated')
        ele.text = "0"
        ele = ET.SubElement(obj, 'difficult')
        ele.text = "0"
        
        x,y = zip(*bb)
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(min(x)))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(max(x)))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(min(y)))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(max(y)))
    
    root = ET.tostring(label)
    root = xml.dom.minidom.parseString(root)
    root = root.toprettyxml(indent='\t')
    root = ET.fromstring(root)
    tree = ET.ElementTree(root)
    tree.write(out_label_path)

def augment(inp_img_path, inp_label_path, out_img_path, out_label_path, seed):
    random.seed(seed)
    
    img = cv2.imread(inp_img_path)
    height, width = img.shape[:2]
    
    label = load_label(inp_label_path)
    bbox = load_bbox(img, label)
    
    while random.random() < .5:
        img = generate_bubble(img, max_alpha=200, exp_base=1.0001, exp_power=1, fill=(72 + random.randint(0, 64),72 + random.randint(0, 64),72))
        
    while random.random() < .5:
        img = generate_bubble(img)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.astype(np.int32)
    img[:,:,1] += random.randint(-30, 30)
    img[:,:,2] += random.randint(-30, 30)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    img = img.astype(np.int32)
    img[:,:,0] += random.randint(-15, 15)
    img[:,:,1] += random.randint(-15, 15)
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    img, bbox = paste_onto_bg(img, bbox)

    cv2.imwrite(out_img_path, img)
    save_label(img, label, bbox, out_label_path)

# random.seed(42)
dir = os.path.join(INP_DIR, 'label')
dir = os.listdir(dir)
dir.sort()

inp_img = []
inp_label = []
out_img = []
out_label = []
seeds = []

fn = lambda fi, seed: os.path.splitext(fi)[0] + '.' + str(seed)
_fn = lambda fi, seed: os.path.splitext(fi)[0]
for fi in dir:
    for seed in range(5):
        inp_img.append(os.path.join(INP_DIR, 'image', _fn(fi, seed) + '.jpg'))
        inp_label.append(os.path.join(INP_DIR, 'label', _fn(fi, seed) + '.xml'))
        out_img.append(os.path.join(OUT_DIR, 'image', fn(fi, seed) + '.jpg'))
        out_label.append(os.path.join(OUT_DIR, 'label', fn(fi, seed) + '.xml'))
        seeds.append(seed)

print(len(inp_img))
t = time.time()
with Pool(100) as pool:
    pool.starmap(augment, zip(inp_img, inp_label, out_img, out_label, seeds))
print(time.time() - t)

# with Pool(5) as pool:
#     img = lambda fi: os.path.splitext(fi)[0] + '.jpg'
#     inp_img = lambda fi: os.path.join(INP_DIR, 'image', img(fi))
#     inp_label = lambda fi: os.path.join(INP_DIR, 'label', fi)
#     out_img = lambda fi: os.path.join(OUT_DIR, 'image', img(fi))
#     out_label = lambda fi: os.path.join(OUT_DIR, 'label', fi)
#     dir = dir[:10]
#     pool.starmap(augment, zip(map(inp_img, dir), map(inp_label, dir), map(out_img, dir), map(out_label, dir)))