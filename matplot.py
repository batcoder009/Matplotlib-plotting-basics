import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 11)
y = x * 2

# plt.plot(x, y)
# plt.title("Line Plot Example")
# plt.xlabel("X values")
# plt.ylabel("Y values")
# plt.show()

x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y)
plt.title("Scatter Plot Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

data = np.random.randn(1000)

plt.hist(data, bins=20)
plt.title("Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

