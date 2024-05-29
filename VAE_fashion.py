import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Lambda, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

#yは正解ラベル
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
image_size = x_train.shape[1]

#都合上、データセットを複製（クラス2から9の抽出用）
x_test_original = x_test.copy()
y_test_original = y_test.copy()

#MNISTデータセットからクラス0と1のみを抽出
x_train = x_train[np.isin(y_train, [0, 1])]
y_train = y_train[np.isin(y_train, [0, 1])]

# クラス2から9のデータをフィルタリング
x_other = x_test_original[np.isin(y_test_original, [2, 3, 4, 5, 6, 7, 8, 9])]
y_other = y_test_original[np.isin(y_test_original, [2, 3, 4, 5, 6, 7, 8, 9])]


#元々(60000, 28, 28)の3次元テンソル→(60000, 28, 28, 1)の4次元テンソルに変換
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_other = np.reshape(x_other,[-1, image_size, image_size, 1])

#各ピクセルの値は0から255の整数値で表現されている(0は黒、255は白を意味する)
#ピクセル値を0から1の範囲に正規化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_other=x_other.astype('float32') / 255

# VAEモデルのパラメータ
#MNIST 画像は 28x28 ピクセルのグレースケール (1チャンネル) なので、入力サイズは (28, 28, 1)
input_shape = (image_size, image_size, 1)
batch_size = 128
latent_dim = 2
epochs = 50

# エンコーダの構築
inputs = Input(shape=input_shape, name='encoder_input')#Input関数でモデルの入力層を定義
x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)#Conv2D は2次元の畳み込み層で、画像データの特徴抽出/カーネルサイズは3x3/28*28から14x14
x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)#14*14から7*7へ/畳み込みフィルター数(カーネル数)は64種類でそれぞれ異なる
x = Flatten()(x)
#Dense(全結合層)で16のユニットを持つ/潜在空間のパラメータを生成するための中間層
x = Dense(16, activation='relu')(x)
#全結合層の出力値(平均と対数分散)の定義
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# サンプリング関数

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))#標準正規分布
    return z_mean + K.exp(0.5 * z_log_var) * epsilon #returnでリパラ

#z_mean と z_log_var を入力として受け取り、定義した"サンプリング関数"を通じて潜在変数 z を生成
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
#Model関数で入力から出力までの計算をカプセル化するモデルオブジェクトを定義←つまりエンコーダのモデル定義
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# デコーダの構築
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')#潜在ベクトル
x = Dense(7 * 7 * 64, activation='relu')(latent_inputs)#潜在ベクトルを受け取り、それを中間層を通じて大きな次元に展開/エンコーダ後の7*7から展開していく
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
decoder_outputs = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
decoder = Model(latent_inputs, decoder_outputs, name='decoder')

# VAEモデルのインスタンス化
outputs = decoder(encoder(inputs)[2])#encoderの潜在変数z
vae = Model(inputs, outputs, name='vae_mlp')

# 損失関数の定義/ReconstructionLossとKLDの損失
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))#K.flatten()で入力と出力の画像をそれぞれ1次元のベクトルに変換/交差エントロピーは0から１値に適する。
reconstruction_loss *= image_size * image_size#画素単位の誤差を画像全体の誤差に変換し、入力画像とデコーダー出力の画像全体の誤差が格納
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)#再構築損失とKLダイバージェンス損失の和の平均を取り、全体のVAE損失を計算
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# モデルのトレーニング
# モデルのトレーニング
vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

other_loss = vae.evaluate(x_other,x_other)
print(f'VAE Other digits loss: {other_loss}')


# エンコーダを使用して潜在変数を抽出
z_mean, _, _ = encoder.predict(x_other)

# デコーダを使用して画像を復号化
decoded_images = decoder.predict(z_mean)

def plot_decoded_images(original, decoded, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 元の画像を表示
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 復号化された画像を表示
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded[i].reshape(28, 28), cmap='gray')
        plt.title("Decoded")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# 10個のサンプルで実際の画像と復号化された画像を表示
plot_decoded_images(x_other, decoded_images)