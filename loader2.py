# -- coding: utf-8 --
import tensorflow as tf
import os

def get_train_data(batch_size):
	# 获取文件名列表，转为tensor list
	filenameslist, filelabelslist = read_list_from_disk()
	filenameslist_tensor = tf.convert_to_tensor(filenameslist, dtype=tf.string)
	filelabelslist_tensor = tf.convert_to_tensor(filelabelslist, dtype=tf.string)

	# 创建队列
	input_queue = tf.train.slice_input_producer(tensor_list=[filenameslist_tensor, filelabelslist_tensor],
                                                  shuffle=False,num_epochs=1)
	trainfile, trainlabel = image_operate(input_queue)

	# 设置小批量数据大小
	files, labels = tf.train.batch(tensors=[trainfile , trainlabel],
                             batch_size=batch_size,
                             num_threads=2)

	# 运行
	sess = tf.Session()
	ini_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	sess.run(ini_op)
	# 启动线程
	coordinator = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
	files,labels = sess.run([files, labels])
	coordinator.request_stop()
	coordinator.join(threads)
	sess.close()
	return tf.minimum(tf.cast(files, tf.float32), 1), tf.minimum(tf.cast(labels, tf.float32), 1) # 将大于0的像素点转化为1

def get_test_data(batch_size):
	# 获取文件名列表，转为tensor list
	filenameslist = read_list_from_disk_test()
	filenameslist_tensor = tf.convert_to_tensor(filenameslist, dtype=tf.string)

	# 创建队列
	input_queue = tf.train.slice_input_producer(tensor_list=[filenameslist_tensor],
                                                  shuffle=False,num_epochs=1)
	testfile = image_operate_test(input_queue)

	# 设置小批量数据大小
	files = tf.train.batch(tensors=[testfile],
                             batch_size=batch_size,
                             num_threads=2)
	return tf.minimum(tf.cast(files, tf.float32), 1) # 将大于0的像素点转化为1

# 定义文件名列表生成函数
def read_list_from_disk():
	filenameslist = []
	filelabelslist = []

	allfiles = [] # 20000个文件夹名称
	data_dir = '/Users/evahuang/Desktop/Github/weather'
	dir_names = [os.path.join(data_dir, 'SRAD2018_TRAIN_00%d' % i) for i in xrange(1, 5)]
	for dir_name in dir_names:
	    for files in os.listdir(dir_name):
	        if (not files == '.DS_Store'):
	        	allfiles.append(dir_name+'/'+files)

	for name in allfiles:
		# 循环20000次，每次产生1条数据
		a61 = []
		for files in os.listdir(name):
			# 循环61次，即61张图片，得到61张图片名字
			if (not files == '.DS_Store'):
				a61.append(name+'/'+files)
		filenameslist.append(a61[1: 31])
		b5 = [a61[35], a61[40], a61[45], a61[50], a61[55], a61[60]]
		filelabelslist.append(b5)

	return filenameslist, filelabelslist

# 定义测试数据文件名列表生成函数
def read_list_from_disk_test():
	filenameslist = []
	dir_name = '/Users/evahuang/Desktop/Github/weather/SRAD2018_Test_1'
	allfiles = os.listdir(dir_name) # 10000个文件夹名称
	if (len(allfiles) > 10000):
		allfiles = allfiles[1:10001]
	for name in allfiles:
		# 循环10000次，每次产生1条数据
		a31 = []
		for files in os.listdir(dir_name+'/'+name):
			# 循环31次，即31张图片，得到31张图片名字
			if (not files == '.DS_Store'):
				a31.append(dir_name+'/'+name+'/'+files)
		filenameslist.append(a31[1: 31])
	return filenameslist

# 文件操作
def image_operate(input_queue):
    # input_queue 为队列，input_queue[0]为数据，input_queue[1]为标签
    labels = []
    for i in range(6):
	    cont_label = tf.read_file(input_queue[1][i])
	    label = tf.image.decode_png(cont_label, 1) # 编码并设置为灰度图片
	    label = tf.reshape(label, [501, 501, 1])
	    labels.append(label)
    labels_res = tf.concat([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]], 1)


    images = []
    for i in range(30):
        cont_image = tf.read_file(input_queue[0][i])
        image = tf.image.decode_png(cont_image, 1)
        image = tf.reshape(image, [501, 501, 1])
        images.append(image)
    images_res = tf.concat([images[0], images[1], images[2], images[3], images[4], images[5], images[6], images[7], images[8], images[9], images[10], images[11], images[12], images[13], images[14], images[15], images[16], images[17], images[18], images[19], images[20], images[21], images[22], images[23], images[24], images[25], images[26], images[27], images[28], images[29]], 1)

    return images_res, labels_res

# 文件操作
def image_operate_test(input_queue):
    # input_queue 为队列，input_queue为数据
    images = []
    for i in range(30):
        cont_image = tf.read_file(input_queue[0][i])
        image = tf.image.decode_png(cont_image, 1)
        image = tf.reshape(image, [501, 501, 1])
        images.append(image)
    images_res = tf.concat([images[0], images[1], images[2], images[3], images[4], images[5], images[6], images[7], images[8], images[9], images[10], images[11], images[12], images[13], images[14], images[15], images[16], images[17], images[18], images[19], images[20], images[21], images[22], images[23], images[24], images[25], images[26], images[27], images[28], images[29]], 1)

    return images_res
