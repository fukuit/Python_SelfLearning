from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 潜在変数の次元
encoding_dim = 32

# 入力用の変数
input_img = Input(shape=(784, ))
# 入力された画像がencodeされたものを格納する変数
encoded = Dense(encoding_dim, activation='relu')(input_img)
# ecnodeされたデータを再構成した画像を格納する変数
decoded = Dense(784, activation='sigmoid')(encoded)
# 入力画像を再構成するModelとして定義
autoencoder = Model(input_img, decoded)

# 入力する画像をencodeする部分
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim, ))
decoder_layer = autoencoder.layers[-1]
# encodeされた画像データを再構成する部分
decoder = Model(encoded_input, decoder_layer(encoded_input))

# AdaDeltaで最適化, loss関数はbinary_crossentropy
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# MNISTデータを前処理する
(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_valid = train_test_split(x_train, test_size=0.175)
x_train = x_train.astype('float32')/255.
x_valid = x_valid.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_valid = x_valid.reshape((len(x_valid), np.prod(x_valid.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# autoencoderの実行
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_valid, x_valid))

# 画像化して確認
encoded_img = encoder.predict(x_test)
decoded_img = decoder.predict(encoded_img)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
