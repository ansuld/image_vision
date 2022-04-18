"""
# 生成yolo索引
# yolo_annotation.py

data/
    classes/
        voc_class.txt/   ---配置文件,保存所有对象信息

    dataset/
        train.txt/       ---yolo训练索引
        test.txt/        ---yolo测试索引
        trainval.txt/    ---yolo训练测试索引
        val.txt/         ---yolo验证索引

"""

import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["hand"]  # 你要训练的对象

# 1.保存所有对象信息
classes_file = open('../data/classes/voc_classes.txt', 'w')
for idx in classes:
    classes_file.write(str(idx))
    classes_file.write('\n')
classes_file.close()


def convert_annotation(year, image_id, list_file):
    """Annotations中xml加上真实框信息
    """
    in_file = open('D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a)
                                        for a in b]) + ',' + str(cls_id))


#wd = getcwd()  # 获得当前的工作目录

for year, image_set in sets:
    image_ids = open('D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' %
                     (year, image_set)).read().strip().split()  # 读取VOC目录下txt中图片路径信息
    # 2.data/dataset目录下yolo索引txt中加上真实框标注信息(model_data)
    list_file = open('../data/dataset/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC%s/JPEGImages/%s.jpg' %
                        (year, image_id))  # 3.VOC目录下voc索引txt中加上图片路径信息
        # 4.Annotations中xml加上真实框信息
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()