import os
import cv2
import random
import image_processing as img_proc
import xml.etree.ElementTree as ET

INP_DIR = '/dataset/crawl/front_cmtnd_resized/'
OUT_DIR = '/dataset/augmented/front_cmtnd_resized/'
BGS_DIR = '/dataset/bg/'

def generate_bubble(img,
    radius=100,
    max_alpha=196,
    exp_base=1.8,
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

def load_bbox(label):
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
        
    return bbox

def paste_onto_bg(img, bbox):
    bgs_dir = os.listdir(BGS_DIR)
    bg_path = os.path.join(BGS_DIR, random.sample(bgs_dir, 1)[0])
    bg = cv2.imread(bg_path)
    
    height, width = img.shape[:2]
    random_factor = (random.random() + 2) / 3
    bg_height, bg_width = [int(random_factor * max(x,y * 1.5)) for x,y in zip(bg.shape[:2], img.shape[:2])]
    bg = cv2.resize(bg, (bg_width, bg_height))
    
    x_offset, y_offset = bg_width - width, bg_height - height
    x_offset, y_offset = [random.randint(0, offset) for offset in [x_offset, y_offset]]
    
    bg[y_offset:y_offset+height, x_offset:x_offset+width] = img
    bbox = {name:[[x[0] + x_offset, x[1] + y_offset] for x in bb] for name, bb in bbox.items()}
    
    return bg, img
        
    

def augment(inp_img_path, inp_label_path, out_img_path, out_label_path):
    img = cv2.imread(inp_img_path)
    height, width = img.shape[:2]
    
    label = load_label(inp_label_path)
    bbox = load_bbox(label)
    
    img = generate_bubble(img, max_alpha=200, exp_base=1.0001, exp_power=1, fill=(72,72,72))
    img = generate_bubble(img)
    
    img, bbox = paste_onto_bg(img, bbox)

    cv2.imwrite(out_img_path, img)

random.seed(42)
augment(os.path.join(INP_DIR, 'image', 'cmnd_nguyen-thi-cuc-994_1600746915.9807026.jpg'),
        os.path.join(INP_DIR, 'label', 'cmnd_nguyen-thi-cuc-994_1600746915.9807026.xml'),
        os.path.join(OUT_DIR, 'image', 'cmnd_nguyen-thi-cuc-994_1600746915.9807026.jpg'),
        '')