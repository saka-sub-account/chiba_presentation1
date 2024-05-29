# 必要なライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

# MNISTデータセットのロード
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']
x = x.values.reshape(-1, 28, 28).astype(np.uint8)
y = y.astype(np.uint8)

# 0, 8, 1の画像を抽出
digits_to_display = [0, 8, 1]
images = []

for digit in digits_to_display:
    index = np.where(y == digit)[0][0]
    images.append(x[index])

# 画像のプロット
fig, axs = plt.subplots(1, 3, figsize=(10, 3))

for i, image in enumerate(images):
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title(f'Digit: {digits_to_display[i]}')
    axs[i].axis('off')

plt.show()
