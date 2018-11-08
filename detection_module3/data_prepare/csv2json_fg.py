import os
import json
import numpy as np


def get_category_id(cls):
    for category in dataset['categories']:
        if category['name'] == cls:
            return category['id']


def get_imgClass_id(cls):
    for i, clas in enumerate(imgClasses):
        if clas == cls:
            return i


def csv2json_fg(label_csv):
    all_data = ['train', 'val']
    classes = ['Pneumonia']

    list = open(label_csv, 'r')
    all_imgs = {}
    for line in list:
        line_split = line.strip().split(',')
        (filename, x, y, w, h, cls) = line_split
        if x not in ['', 'x']:
            if filename not in all_imgs:
                all_imgs[filename] = {}
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = 1024
                all_imgs[filename]['height'] = 1024
                all_imgs[filename]['bboxes'] = []
                all_imgs[filename]['cls'] = int(cls)

            all_imgs[filename]['bboxes'].append(
                {'class': 'Pneumonia', 'x1': float(x), 'y1': float(y), 'w': float(w), 'h': float(h)})

    for Dataset in all_data:
        dataset = {
            'licenses': [],
            'info': {},
            'images': [],
            'annotations': [],
            'categories': []
        }

        for i, cls in enumerate(classes, 1):
            dataset['categories'].append({
                'id': i,
                'name': cls,
                'supercategory': 'kaggle'
            })

        j = 0
        ii = 0
        for dd in all_imgs:
            ii += 1
            if Dataset == 'train':
                if ii > 5600:
                    continue
            if Dataset == 'val':
                if ii <= 5600:
                    continue
            fname = all_imgs[dd]['filepath']
            width = all_imgs[dd]['width']
            height = all_imgs[dd]['height']

            dataset['images'].append({
                'coco_url': '',
                'date_captured': '',
                'file_name': fname + '.png',
                'flickr_url': '',
                'id': ii,
                'license': 0,
                'width': width,
                'height': height,
                'imgClass': all_imgs[dd]['cls'],
            })

            bboxes = all_imgs[dd]['bboxes']
            for bbox in bboxes:
                j += 1
                x1 = bbox['x1']
                y1 = bbox['y1']
                w = bbox['w']
                h = bbox['h']
                box_width = max(1., w)
                box_height = max(1., h)

                dataset['annotations'].append({
                    'area': box_width * box_height,
                    'bbox': [x1, y1, box_width, box_height],
                    'category_id': 0,
                    'id': j,
                    'image_id': ii,
                    'iscrowd': 0,
                    'segmentation': []
                })

        if Dataset == 'train':
            print('train imgs:', 5600, 'total_bboxes:', j)
        if Dataset == 'val':
            print('val imgs:', ii - 5600, 'total_bboxes:', j)

        folder = './annotations'
        if not os.path.exists(folder):
            os.makedirs(folder)
        json_name = '{}/{}_pos.json'.format(folder, Dataset)
        with open(json_name, 'w') as f:
            json.dump(dataset, f)



