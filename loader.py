# -- coding: utf-8 --
import tensorflow as tf
import os
import cv2
import imghdr
from PIL import Image
from PIL import ImageFile
import imghdr

def get_train_data(batch_size, index, filenameslist, filelabelslist):
	nameslist = filenameslist[batch_size * index:batch_size * index+batch_size]
	labelslist = filelabelslist[batch_size * index:batch_size * index+batch_size]

	images = []
	labels = []
	for i in range(batch_size):
		images.append(image_operate([nameslist[i], labelslist[i]])[0])
		labels.append(image_operate([nameslist[i], labelslist[i]])[1])

	images = tf.minimum(tf.cast(images, tf.float32), 1) # 将大于0的像素点转化为1
	labels = tf.minimum(tf.cast(labels, tf.float32), 1) # 将大于0的像素点转化为1
	return tf.reshape(images, [batch_size, 501, 15030, 1]), tf.reshape(labels, [batch_size, 501, 3006, 1])


# 文件操作
def image_operate(input_queue):
    # input_queue 为队列，input_queue[0]为数据，input_queue[1]为标签
    labels = []
    for i in range(6):
	    cont_label = cv2.imread(input_queue[1][i], cv2.IMREAD_GRAYSCALE)
	    labels.append(cont_label)
    labels_res = tf.concat([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]], 1)


    images = []
    for i in range(30):
        cont_image = cv2.imread(input_queue[0][i], cv2.IMREAD_GRAYSCALE)
        images.append(cont_image)
    images_res = tf.concat([images[0], images[1], images[2], images[3], images[4], images[5], images[6], images[7], images[8], images[9], images[10], images[11], images[12], images[13], images[14], images[15], images[16], images[17], images[18], images[19], images[20], images[21], images[22], images[23], images[24], images[25], images[26], images[27], images[28], images[29]], 1)

    return images_res, labels_res

def get_test_data(filenameslist, batch_size, index):
	nameslist = filenameslist[batch_size * index:batch_size * index+batch_size]
	images = []
	for i in range(batch_size):
		images.append(image_operate_test(nameslist[i]))

	images = tf.minimum(tf.cast(images, tf.float32), 1) # 将大于0的像素点转化为1
	return tf.reshape(images, [batch_size, 501, 15030, 1])


# 文件操作
def image_operate_test(inputs):
    images = []
    for i in range(30):
        name = inputs[i]
        # cont_image = cv2.imread(inputs[i], cv2.IMREAD_GRAYSCALE)
        # ImageFile.LOAD_TRUNCATED_IMAGES = True
        # if imghdr.what(name) == "png":
        #     Image.open(name).convert("RGB").save(name)
        cont_image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        images.append(cont_image)
    images_res = tf.concat([images[0], images[1], images[2], images[3], images[4], images[5], images[6], images[7], images[8], images[9], images[10], images[11], images[12], images[13], images[14], images[15], images[16], images[17], images[18], images[19], images[20], images[21], images[22], images[23], images[24], images[25], images[26], images[27], images[28], images[29]], 1)

    return images_res
