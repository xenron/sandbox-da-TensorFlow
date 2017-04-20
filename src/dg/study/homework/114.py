import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
# y = tf.placeholder(tf.float32,[None,10])
y_ = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

# l1 = addLayer(x,784,100,activity_function=tf.nn.relu)
# l2 = addLayer(l1,10,10,activity_function=None)

prediction = tf.nn.softmax(tf.matmul(x,W)+b)
# prediction = tf.matmul(x,W)+b

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        
        correct_prediction = tf.equal(tf.arg_max(y,1), tf.argmax(y_,1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print sess.run(acc,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
        
        # acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        # print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))

==================================================


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784 
OUTPUT_NODE = 10 

LAYER1_NODE = 500
BATCH_SIZE = 100 
LEARNING_RATE = 0.01
TRAINING_STEPS = 20000

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.0, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE]))
    layer1 = tf.nn.relu(tf.matmul(x, weights1) + bias1)
    y = tf.matmul(layer1, weights2) + bias2
    global_step = tf.Variable(0, trainable=False)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    loss = tf.reduce_mean(cross_entropy)
    train_op=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}     

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})


        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))
 
def main(argv=None): 

    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
