#!/usr/bin/env python
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import skimage
import skimage.io
import os


# 将RGB按照BGR重新组装，然后对每一个RGB对应的值减去一定阈值
def prepare_image(image):
    H, W, _ = image.shape
    h, w = (img_width, img_height)

    h_off = max((H - h) // 2, 0)
    w_off = max((W - w) // 2, 0)
    image = image[h_off:h_off + h, w_off:w_off + w, :]

    image = image[:, :, :: -1]

    image = image.astype(np.float32, copy=False)
    image = image * 255.0
    image = image - np.array(VGG_MEAN, dtype=np.float32)

    image = np.expand_dims(image, axis=0)
    return image


# 使用TFLite文件检测
def getResultFromFilePathByTFLite(path):
    model_path = "./model/nsfw.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    # print(str(input_details))
    output_details = interpreter.get_output_details()
    # print(str(output_details))

    im = Image.open(path)
    # im = Image.open(r"./images/image1.png")
    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize((256, 256), resample=Image.BILINEAR)
    fh_im = io.BytesIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)

    image = (skimage.img_as_float(skimage.io.imread(fh_im, as_gray=False))
             .astype(np.float32))

    # 填装数据
    final = prepare_image(image)
    interpreter.set_tensor(input_details[0]['index'], final)

    # 调用模型
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 出来的结果去掉没用的维度
    result = np.squeeze(output_data)
    print('TFLite-->>result:{},path:{}'.format(result, path))
    return result


def getResultListFromDir():
    img_dir = './test_images'
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        if img != ".DS_Store" and img != ".localized":
            getResultFromFilePathByTFLite(img_path)


if __name__ == "__main__":
    VGG_MEAN = [104, 117, 123]

    img_width, img_height = 224, 224

    #检测加载Downloads下所有文件，逐个输出检测结果
    getResultListFromDir()
