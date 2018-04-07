import tensorflow as tf
from func import *

# 通过numpy工具包生成模拟数据集
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络参数
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# 在 shape 的一个维度上使用None可以方便使用不同的 batch 大小。在训练时需要把数据分成比较小的 batch，但是在测试时，可以一次性使用全部的数据。
# 当数据集比较小时这样比较方便测试，但数据集比较大时，将大量数据放入一个 batch 可能会导致内存溢出
x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")

# 定义神经网络前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# 定义损失函数和反向传播算法
y = tf.sigmoid(y)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)) + (1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

# 定义规则来给出样本的标签。在这里所有x1+x2<1的样例被认为是正样本，其他样本为负样本。使用0表示负样本，1表示正样本。
Y = [ [int(x1+x2<1)] for (x1,x2) in X]


# 测试数据
# x_text = tf.constant([
#         [1,0.5],
#         [0.01,0.01],
#         [8,8],
#         [0.08,0.38],
#         [38,38],
#         [38,328],
#         [0.32,0.1],
#         [0.3,0.8],
#
#     ],name="x_text")

x_text = [
        [1,0.5],
        [3.01,0.01],
        [8,8],
        [6.08,0.38],
        [0.38,0.1],
        [0.23,0.01],
        [0.12,0.1],
        [0.01,0.8],

    ]

def predict(x,w1,w2):

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # return tf.sigmoid(y)
    return y

# 创建一个会话来运行Tensorflow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)

    # print("训练前神经网络参数：")
    # print(sess.run(w1))
    # print(sess.run(w2))

    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取 batch_size 个样本进行训练。
        # print("第",i,"次迭代")
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        # print("start: ",start)
        # print("end: ",end)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        if i % 2500 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross entropy on all data is %g" % (i,total_cross_entropy))

    # print("训练后神经网络参数：")
    # print(sess.run(w1))
    # print(sess.run(w2))


    print("测试数据")
    print(x_text)
    # print(sess.run(x_text))
    print("预测结果")
    print(sess.run(predict(x_text,w1,w2)))

    # index = 1
    # print("第一组数据")
    # print(X[index])
    # print("第一组数据标签")
    # print(Y[index])
    # print("预测结果")
    # print(sess.run(predict([[X[index][0],X[index][1]]], w1, w2)))


















