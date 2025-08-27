import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

# Effect of bias
plt.figure(figsize=(7, 4))
w = 1
for b in [-5, 0, 5]:
    y = w * x + b
    plt.plot(x, y, label=f'b={b}')
plt.title('Effect of Bias (y = wx + b)')
plt.legend()
plt.grid(True)
plt.savefig("bias_backprop.png")
plt.show()

# Effect of weight
plt.figure(figsize=(7, 4))
b = 0
for w in [-3, -1, 0, 1, 3]:
    y = w * x + b
    plt.plot(x, y, label=f'w={w}')
plt.title('Effect of Weight (y = wx + b)')
plt.legend()
plt.grid(True)
plt.savefig("weight_backprop.png")
plt.show()
