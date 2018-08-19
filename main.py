import base64
import io
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, url_for
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image, ImageFilter

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_to_be_tested = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]), name='Weight')  # weight
b = tf.Variable(tf.zeros([10]))  # bias

y = tf.nn.softmax(tf.matmul(x, W) + b)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)


def train():
  print('training start')
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(
      y_ * tf.log(y), reduction_indices=[1]))

  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  print('training ended')


def test_image(raw_image):
  raw_image_32 = [np.float(raw_image_p) for raw_image_p in raw_image]
  np_img = np.array(raw_image_32)
  result = tf.nn.softmax(tf.matmul(x_to_be_tested, W) + b)
  final_result = tf.argmax(result, 1)
  py_res = sess.run(final_result, feed_dict={x_to_be_tested: [np_img]})[0]

  return py_res


app = Flask(__name__)
train()

@app.route('/')
def index():
  return "hello world"


@app.route('/api', methods=['POST'])
def upload_img():
  img_str = request.form['img'][22:]
  img_data = base64.b64decode(img_str)
  img_obj = Image.open(io.BytesIO(img_data))
  img_obj = img_obj.resize((28, 28))
  img_pix = img_obj.load()
  img_arr = []

  for index_b in range(28):
    for index_a in range(28):
      pix_l = img_pix[index_a, index_b][3]/255
      img_arr.append(pix_l)

  num = test_image(img_arr)
  num = num.astype(np.float)

  return jsonify(response=1, number=num)


if __name__ == '__main__':
  app.run()
