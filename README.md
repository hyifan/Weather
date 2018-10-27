### 赛题：
IEEE ICDM 2018 全球气象AI挑战赛

### 链接：
https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.86e033afH3NY8r&raceId=231662



### 算法思路：
1、数据处理：每条训练数据由61张501*501的图片，选取了前三个小时每隔6分钟30张图片组成的501*(501*30)为输入，后三个小时每隔30分钟6张图片组成的501*(501*6)为label<br/>
2、选用激活函数为ReLU函数<br/>
3、选用的代价函数为二次代价函数<br/>
4、选用算法为CNN，用tensorflow实现，网络深度为：卷积层 -  - 池化层 - LRN层 - 全链接层<br/>
5、首先用get_params()优化代价函数并获取最终训练的参数w和b，然后使用get_result(w,b)获取最终结果的数组形式，最后用array_to_png()将数组转成最终图片

### 在终端运行的指令：
``` python
import tf_cnn
w1, b1, w2, b2 = tf_cnn.get_params(10, 10)
logits = tf_cnn.get_result(w1, b1, w2, b2, 0, 10, 10)
tf_cnn.array_to_png(logits, 0, 100)
```