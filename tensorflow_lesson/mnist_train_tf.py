import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNISTデータのダウンロードと定義
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# パラメーター定義
# 学習用画像
x = tf.placeholder(tf.float32, [None, 784])
# weight
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))

# 目的関数
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 学習のための準備
# 正解データ
y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy: 最適化パラメーター
#cross_entropy = tf.reduce_mean(
#    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# 最急降下法による最適化パラメーターの最小化
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# tensorflow 0.12以降はinitialize_all_variables()は使えない
# init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

# 1000回のiterationを実施
sess = tf.Session()
sess.run(init)
for i in range(1000):
    # train.next_batch(100)で100個分の画像データとラベルを取得(ミニバッチ)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # train_stepに、テストデータとしてのbatch_xsとその正解batch_ysを入力
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 正解率の表示
correct_prediction = tf.equal(
    tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
