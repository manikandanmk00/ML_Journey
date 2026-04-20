import numpy as np

# 1. Arrays — foundation of everything
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape)   # (5,)
print(b.shape)   # (2, 3)

# 2. Matrix operations — this IS neural networks
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(np.dot(A, B))      # Matrix multiplication
print(A.T)               # Transpose
print(np.sum(A, axis=0)) # Sum along axis

# 3. Broadcasting — very important in ML
x = np.array([1, 2, 3])
print(x * 2)      # [2, 4, 6]
print(x + 10)     # [11, 12, 13]

# 4. Random — used everywhere in ML
weights = np.random.randn(3, 3)  # Random matrix
print(weights)



# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 1*4 + 2*5 + 3*6 = 32

# Matrix multiplication — THIS IS HOW NEURAL NETWORKS WORK
# Input data: 3 samples, 4 features
X = np.random.randn(3, 4)

# Weights: 4 inputs → 2 outputs
W = np.random.randn(4, 2)

# Forward pass of a neural network layer!
output = np.dot(X, W)
print(output.shape)  # (3, 2)
print(output)

# GRADIENTS — You know calculus!
# f(x) = x^2
# f'(x) = 2x  ← this is a gradient
# In neural networks we compute gradients to learn
x = np.array([1.0, 2.0, 3.0])
gradient = 2 * x
print(gradient)  # [2. 4. 6.]