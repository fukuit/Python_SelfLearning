'''
Convolutional Variational AutoEncoder
'''
from keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.losses import binary_crossentropy

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

# MNISTデータを前処理する
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.175)
# 正規化する
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train), image_size, image_size, 1))
x_test = np.reshape(x_test, (len(x_test), image_size, image_size, 1))

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.175)

# network parameters
input_shape = (image_size, image_size, 1)
batch_size = 64
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 50

# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

shape = K.int_shape(x)

x = Flatten()(x)
x = Dense(16, activation='relu')(x)

# 潜在変数
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# autoencoderの定義
outputs = decoder(encoder(inputs)[2])
autoencoder = Model(inputs, outputs, name='vae')

models = (encoder, decoder)
data = (x_test, y_test)

# loss関数
# Compute VAE loss
reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                          K.flatten(outputs))
reconstruction_loss *= image_size * image_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)

autoencoder.add_loss(vae_loss)
autoencoder.compile(optimizer='adam')
autoencoder.summary()

# 学習に使うデータを1に限定する
x_train = x_train[y_train==1]
x_valid = x_valid[y_valid==1]

# autoencoderの実行
autoencoder.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_valid, None))

# 実行結果の表示
n = 10
decoded_imgs = autoencoder.predict(x_test[:n])

plt.figure(figsize=(10, 4))
for i in range(n):
    # original_image
    orig_img = x_test[i].reshape(image_size, image_size)

    # reconstructed_image
    reconst_img = decoded_imgs[i].reshape(image_size, image_size)

    # diff image
    diff_img = ((orig_img - reconst_img)+2)/4
    diff_img = (diff_img*255).astype(np.uint8)
    orig_img = (orig_img*255).astype(np.uint8)
    reconst_img = (reconst_img*255).astype(np.uint8)
    diff_img_color = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)

    # display original
    ax = plt.subplot(3, n,  i + 1)
    plt.imshow(orig_img, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(reconst_img, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display diff
    ax = plt.subplot(3, n, i + n*2 + 1)
    plt.imshow(diff_img, cmap=plt.cm.jet)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 学習結果の保存
autoencoder.save('./ae_mnist.h5')

# json and weights
model_json = autoencoder.to_json()
with open('ae_mnist.json', 'w') as json_file:
    json_file.write(model_json)
autoencoder.save_weights('ae_mnist_weights.h5')

## load model
#from keras.models import model_from_json
#
## load model.json
#json_file = open('vcae_mnist.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#
#loaded_model = model_from_json(loaded_model_json)
#
## load weights into new model
#loaded_model.load_weights('vcae_mnist_weights.h5')
#
## Compute VAE loss
#xent_loss = K.mean(metrics.binary_crossentropy(input_img, decoded), axis=-1)
#kl_loss =  - 0.5 * K.mean(K.sum(1 + K.log(K.square(z_log_var)) - K.square(z_mean) - K.square(z_log_var), axis=-1))
#vae_loss = K.mean(xent_loss + kl_loss)
#
#loaded_model.add_loss(vae_loss)
#loaded_model.compile(optimizer=opt)
