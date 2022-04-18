from tensorflow.keras.layers import Input
from yolo import YOLO,detect_video
from PIL import Image

def for_img(yolo):
    path = 'D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC2007/keras-yolo3-master/docs/test.jpg'
    try:
        image = Image.open(path)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
    yolo.close_session()


def for_video(yolo):
    detect_video(yolo, "D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC2007/keras-yolo3-master/docs/xuanya.mp4", "D:/pythonstudy/jupyter-notebook/image_vision/VOCdevkit/VOC2007/keras-yolo3-master/docs/xuanya_detect.mp4")


if __name__ == '__main__':
    _yolo = YOLO()

    # for_img(_yolo)
    for_video(_yolo)
