import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ランダムシードの設定
np.random.seed(42)

# 初期のガウス分布
mu_initial = 0
sigma_initial = 1
x = np.linspace(-5, 5, 1000)
y_initial = norm.pdf(x, mu_initial, sigma_initial)

# 深層ガウス過程による変換（仮想的な例）
# シンプルな非線形変換を適用して複雑な分布を生成
def deep_gp_transform(x):
    y1 = np.tanh(x)
    y2 = np.sin(y1 * 3)
    y3 = y2 + 0.1 * np.random.randn(*y2.shape)
    return y3

# 変換後の複雑な分布
y_transformed = deep_gp_transform(x)

# 可視化
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# 初期のガウス分布
axes[0].plot(x, y_initial, color='black')
axes[0].set_title("Initial Gaussian Distribution")
axes[0].set_xlim([-5, 5])
axes[0].set_ylim([0, 0.5])
axes[0].axis('off')

# 変換後の複雑な分布
axes[1].plot(x, y_transformed, color='black')
axes[1].set_title("Transformed Complex Distribution")
axes[1].set_xlim([-5, 5])
axes[1].set_ylim([-1, 1])
axes[1].axis('off')

plt.tight_layout()
plt.show()
