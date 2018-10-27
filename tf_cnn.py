# -- coding: utf-8 --
import loader
import tensorflow as tf
import os
from PIL import Image

# 执行顺序
# w1, b1, w2, b2 = tf_cnn.get_params(10, 10)
# logits = tf_cnn.get_result(w1, b1, w2, b2, 0, 10, 10)
# tf_cnn.array_to_png(logits, 0, 100)

def get_params(max_steps, batch_size):
	filenameslist, filelabelslist = read_list_from_disk()

	image_holder = tf.placeholder(tf.float32, [batch_size, 501, 15030, 1])
	label_holder = tf.placeholder(tf.float32, [batch_size, 501, 3006, 1])

	# 卷积层 - ReLU - 池化层 - LRN层
	weight1 = variable_with_weight_loss(shape=[5, 5, 1, 16], stddev=5e-2, wl=0.0) # 第一层卷积核之共享权重
	kernel1 = tf.nn.conv2d(image_holder, weight1, [1,1,1,1], padding='SAME') # 对输入图像进行卷积操作
	bias1 = tf.Variable(tf.constant(0.0, shape=[16])) # 初始化第一层卷积核之共享偏置
	conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1)) # 第一层卷积层输出值，激活函数为ReLU
	pool1 = tf.nn.max_pool(conv1, ksize=[1,3,5,1], strides=[1,1,5,1], padding='SAME') # 将结果经过混合层处理，混合层选用最大池化层，尺寸为[3,5]，跨距为[1,5]
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75) # LRN层处理

	reshape = tf.reshape(norm1, [batch_size * 501, -1])
	dim = reshape.get_shape()[1].value # reshape第一维数量

	# 设置输出层
	weight2 = variable_with_weight_loss(shape=[dim, 3006], stddev=0.04, wl=0)
	bias2 = tf.Variable(tf.constant(0.1, shape=[3006]))
	logits = tf.nn.relu(tf.matmul(reshape, weight2) + bias2)
	logits = tf.reshape(logits, [batch_size, 501, 3006, 1])

	# 计算代价函数
	loss = loss_fun(logits, label_holder)
	train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) # 应用Adam优化算法

	# 建立会话
	sess = tf.Session()
	ini_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	sess.run(ini_op)

	# 运行
	for step in range(max_steps):
		images_train, labels_train = loader.get_train_data(batch_size, step, filenameslist, filelabelslist)
		image_batch, label_batch = sess.run([images_train, labels_train])
		_, loss_value, w1, b1, w2, b2 = sess.run([train_op, loss, weight1, bias1, weight2, bias2], feed_dict={image_holder: image_batch, label_holder:label_batch})
		print(('step %d, loss=%.2f') % (step, loss_value))

	sess.close()

	return w1, b1, w2, b2

def get_result(weight1, bias1, weight2, bias2, start_step, end_step, batch_size):
	filenameslist = read_list_from_disk_test()
	types = FOR_SIX()

	image_holder = tf.placeholder(tf.float32, [batch_size, 501, 15030, 1])

	# 卷积层 - ReLU - 池化层 - LRN层
	kernel1 = tf.nn.conv2d(image_holder, weight1, [1,1,1,1], padding='SAME') # 对输入图像进行卷积操作
	conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1)) # 第一层卷积层输出值，激活函数为ReLU
	pool1 = tf.nn.max_pool(conv1, ksize=[1,3,5,1], strides=[1,1,5,1], padding='SAME') # 将结果经过混合层处理，混合层选用最大池化层，尺寸为[3,5]，跨距为[1,5]
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75) # LRN层处理
	reshape = tf.reshape(norm1, [batch_size * 501, -1])
	dim = reshape.get_shape()[1].value # reshape第一维数量

	# 设置输出层
	logits = tf.nn.relu(tf.matmul(reshape, weight2) + bias2)
	logits = tf.reshape(logits, [batch_size, 501, 3006])

	answers = []
	for i in range(batch_size):
		answers.append(handle_for_six(logits[i], types))

	# 建立会话
	sess = tf.Session()
	ini_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	sess.run(ini_op)

	res = []
	for step in range(start_step, end_step):
		images_test = loader.get_test_data(filenameslist, batch_size, step)
		image_batch = sess.run(images_test)
		ans = sess.run(answers, feed_dict={image_holder: image_batch})
		res.extend(ans)
		print(('step %d') % (step))

	sess.close()

	return res

def array_to_png(logits, start_step, end_step):
    testFiles = read_test_filesname(start_step, end_step)
    i = 0
    for logit in logits:
        for data in logit:
            data[data > 0] = 255
            data = data.astype('uint8')
            img = Image.fromarray(data)
            if (not os.path.exists('res/'+ testFiles[i/6])):
                os.mkdir('res/'+ testFiles[i/6])
            img.save('res/'+ testFiles[i/6] + '/' + testFiles[i/6] + '_f00' + str(i%6 + 1) +'.png')
            new_im = Image.fromarray(data)
            i += 1

# variable_with_weight_loss函数创建网络层的参数并初始化，shape=[核长，核宽，颜色通道，核数量]
def variable_with_weight_loss(shape, stddev, wl):
	# wl 即L2规范中lamda值
	# 初始化正太分布的w
	var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
	if wl is not None:
		# 计算sum(w**2)*(lamda/2)
		weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
		tf.add_to_collection('losses', weight_loss)
	return var

# 二次代价函数
def loss_fun(logits, labels):
	loss_mean = tf.losses.mean_squared_error(logits, labels)
	tf.add_to_collection('losses', loss_mean)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def FOR_SIX():
	a = []
	b = []
	c = []
	d = []
	e = []
	f = []
	g = []
	for i in range(501):
		a.append(0)
		b.append(1)
		c.append(2)
		d.append(3)
		e.append(4)
		f.append(5)
	a.extend(b)
	a.extend(c)
	a.extend(d)
	a.extend(e)
	a.extend(f)
	for i in range(501):
		g.append(a)
	return g

# 将每条数据分为6张图片
def handle_for_six(logit, types):
	logits = tf.dynamic_partition(logit, types, 6)
	answers = []
	for i in range(6):
		res = tf.rint(logits[i])
		res = tf.maximum(res, 0)
		res = tf.minimum(res, 1)
		res = tf.reshape(res, [501, 501])
		answers.append(res)
	return answers

# 定义训练数据文件名列表生成函数
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

# 定义最后保存的10000条数据名称
def read_test_filesname(start_step, end_step):
	filenameslist = []
	dir_name = '/Users/evahuang/Desktop/Github/weather/SRAD2018_Test_1'
	allfiles = os.listdir(dir_name) # 10000个文件夹名称
	if (len(allfiles) > 10000):
		allfiles = allfiles[start_step+1:end_step+1]
	else:
		allfiles = allfiles[start_step:end_step]
	return allfiles


# 学习笔记
# tf.Variable(initializer,name):定义变量，initializer是初始化参数，可以有tf.random_normal，tf.constant等，name就是变量的名字
# tf.constant():定义常量
# tf.truncated_normal(shape, mean, stddev):shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。
# tf.multiply(x, y, name=None):x与y相乘
# tf.nn.l2_loss([t, t, t]) = sum(t ** 2) / 2
# tf.nn.relu(features, name=None):relu激活函数，f(x)=max(0,x)，即输入的x的tensor所有的元素中如果小于零的就取零
# tf.nn.bias_add(value, bias, name=None):将bias加到value，其中bias必须是一维
# tf.reshape(tensor,shape, name=None): 将tensor变换为参数shape的形式。
# tf.cast(x, dtype, name=None):用于改变某个张量的数据类型
# tf.reduce_mean():对向量求均值
# tf.nn.in_top_k(predictions, targets, k, name=None):第一个参数是train data，第二个是labels，k一般取1。若train data最大值与对应labels一致，返回true，否则false
# tf.InteractiveSession():与 tf.Session 的作用相同，只不过 tf.InteractiveSession 在建立时，会自动将自己设定为预设的 session，这样可以让我們少打一些字。
# tf.global_variables_initializer():调用 tf.global_variables_initializer 仅会创建并返回 TensorFlow 操作的句柄。当我们使用 tf.Session.run 运行该操作时，该操作将初始化所有全局变量。
# tf.train.start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection=tf.GraphKeys.QUEUE_RUNNERS):使用之后，启动填充队列的线程，这时系统就不再“停滞”。此后计算单元可以拿到数据并进行计算

'''
tf.add_to_collection:把变量放入一个集合，把很多变量变成一个列表
tf.get_collection:从一个结合中取出全部变量，是一个列表
tf.add_n:把一个列表的东西都依次加起来
'''

'''
"SAME" = with zero padding
"VALID" = without padding

tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None
)
1.input是一个4d输入[batch_size, in_height, in_width, n_channels]，表示图片的批数，大小和通道
2.filter是一个4d输入[filter_height, filter_width, in_channels, out_channels]，表示kernel的大小，输入通道数和输出通道数，其中输出通道数表示从上一层提取多少特征
3.strides是一个1d输入，长度为4，其中stride[0]和stride[3]必须为1，一般格式为[1, stride[1], stride[2], 1]，在大部分情况下，因为在height和width上的步进设为一样，因此通常为[1, stride, stride, 1]
4.padding是一个字符串输入，分为SAME和VALID分别表示是否需要填充，因为卷积完之后因为周围的像素没有卷积到，因此一般是会出现卷积完的输出尺寸小于输入的现象的，这时候可以利用填充
返回一个Tensor，这个输出，就是我们常说的feature map

tf.nn.max_pool(
	value,
	ksize,
	strides,
	padding,
	name=None
)
1.value是需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
2.ksize是池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
3.strides和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride,1]
4.padding和卷积类似，可以取'VALID' 或者'SAME'
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式

tf.nn.lrn(
	input,
	depth_radius,
	bias,
	alpha,
	beta
)
sqr_sum[a, b, c, d] = sum(input[a,b, c, d - depth_radius : d + depth_radius + 1] ** 2)
output = input / (bias +alpha * sqr_sum) ** beta

tf.nn.sparse_softmax_cross_entropy_with_logits(
	_sentinel=None,
	labels=None,
	logits=None,
	name=None
)
1.labels: labels的每一行为真实类别的索引，形状[batch_size]
2.logits: 未缩放的对数概率，这个操作的输入logits是未经缩放的，该操作内部会对logits使用softmax操作，形状 [batch_size, num_classes]
3.dims: 类的维度，默认-1，也就是最后一维
4.name: 该操作的名称
返回值：长度为batch_size的一维Tensor, 和label的形状相同，和logits的类型相同
为了追求速度，把原来的神经网络输出层的softmax和cross_entrop何在一起计算
ci = y*lna + (1-y)*ln(1-a) ; a = softmax(logits) ; y = labels


tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam'
)
1.learning_rate：Tensor或浮点值。学习率。
2.beta1：浮点值或常量浮点张量。第一时刻的指数衰减率估计。
3.beta2：浮点值或常量浮点张量。第二时刻的指数衰减率估计。
4.epsilon：数值稳定性的一个小常数，防止除零操作。
5.use_locking：如果True使用锁定进行更新操作。
6.name：应用渐变时创建的操作的可选名称。默认为“Adam”。
Adam优化算法

minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name=None,
    grad_loss=None
)
1.loss：包含要最小化的值的Tensor。
2.global_step：可选变量在变量更新后递增1的变量。
3.var_list：要更新的Variable对象的可选列表或元组，以最大限度地减少损失。默认为在GraphKeys.TRAINABLE_VARIABLES键下的图表中收集的变量列表。
4.gate_gradients：如何控制渐变的计算。可以是GATE_NONE，GATE_OP或GATE_GRAPH。
5.aggregation_method：指定用于组合渐变项的方法。有效值在AggregationMethod类中定义。
6.colocate_gradients_with_ops：如果为True，请尝试将渐变与相应的op进行对齐。
7.name：返回操作的可选名称。
8.grad_loss：可选。一个Tensor持有为损失计算的梯度。
返回一个更新var_list中变量的操作。如果global_step不是None，则该操作也会增加global_step。
'''