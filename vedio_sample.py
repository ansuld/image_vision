import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 参数
yolo_dir = 'D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC2007'  # YOLO文件路径
CONFIDENCE = 0.5  # 过滤弱检测的最小概率
THRESHOLD = 0.4  # 非最大值抑制阈值


def yolov3_vedio(img, yolo_dir, CONFIDENCE, THRESHOLD):
    weightsPath = os.path.join(yolo_dir, 'keras-yolo3-master/yolov3.weights')  # 权重文件
    configPath = os.path.join(yolo_dir, 'keras-yolo3-master/yolov3.cfg')  # 配置文件
    labelsPath = os.path.join(yolo_dir, 'keras-yolo3-master/data/classes/coco.name')  # label名称
    # 加载网络、配置权重
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)  ## 利用下载的文件
    print("[INFO] loading YOLO from disk...")  ## 可以打印下信息

    blobImg = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), None, True,
                                   False)  ## net需要的输入是blob格式的，用blobFromImage这个函数来转格式
    net.setInput(blobImg)  ## 调用setInput函数将图片送入输入层

    # 获取网络输出层信息（所有输出层的名字），设定并前向传播
    outInfo = net.getUnconnectedOutLayersNames()  ## 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
    start = time.time()
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))  ## 可以打印下信息

    # 拿到图片尺寸
    (H, W) = img.shape[:2]
    # 过滤layerOutputs
    # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
    # 过滤后的结果放入：
    boxes = []  # 所有边界框（各层结果放一起）
    confidences = []  # 所有置信度
    classIDs = []  # 所有分类ID

    # # 1）过滤掉置信度低的框框
    for out in layerOutputs:  # 各个输出层
        for detection in out:  # 各个框框
            # 拿到置信度
            scores = detection[5:]  # 各个类别的置信度
            classID = np.argmax(scores)  # 最高置信度的id即为分类id
            confidence = scores[classID]  # 拿到置信度

            # 根据置信度筛查
            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)  # boxes中，保留的box的索引index存入idxs

    # 得到labels列表
    with open(labelsPath, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')

    # 应用检测结果
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3),
                               dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
    if len(idxs) > 0:
        for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            print(COLORS,idxs.flatten(),classIDs)
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            print("[INFO] predect result is: ", text)
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                       2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px

    return img


cap = cv2.VideoCapture('D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC2007/keras-yolo3-master/docs/xuanya.mp4')
#"D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC2007/keras-yolo3-master/docs/xuanya.mp4",
outpath="D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC2007/keras-yolo3-master/docs/xuanya_detect.mp4"
video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out=cv2.VideoWriter(outpath, video_FourCC, video_fps, video_size)
while (True):
    ret, frame = cap.read()  # 读取帧
    if ret == True:
        img=yolov3_vedio(frame, yolo_dir, CONFIDENCE, THRESHOLD)
        cv2.imshow('frame', frame)
    # else:
    #     cap.release()
    #     cv2.waitKey(0)
    out.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按‘q’退出
        break


# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()


