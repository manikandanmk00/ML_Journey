# ============================================
# FULL LINEAR REGRESSION TRAINING FROM SCRATCH
# ============================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Create Data
X = np.random.randn(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.5

# 2. Initialize weights
W = np.random.randn(1, 1)
b = np.zeros((1, 1))

# 3. Training settings
learning_rate = 0.01
epochs = 1000  # How many times we loop through data
loss_history = []

print("Starting Training...")
print(f"True W should be: 3.0, True b should be: 2.0")
print(f"Initial W: {W[0][0]:.4f}, Initial b: {b[0][0]:.4f}")
print("-" * 50)

# 4. Training Loop — THE HEART OF ML
for epoch in range(epochs):
    
    # Step 1: Forward pass
    y_pred = np.dot(X, W) + b
    
    # Step 2: Compute loss
    n = len(y)
    loss = (1/n) * np.sum((y - y_pred)**2)
    loss_history.append(loss)
    
    # Step 3: Compute gradients
    error = y_pred - y
    dW = (2/n) * np.dot(X.T, error)
    db = (2/n) * np.sum(error)
    
    # Step 4: Update weights
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | W: {W[0][0]:.4f} | b: {b[0][0]:.4f}")

print("-" * 50)
print(f"Final W: {W[0][0]:.4f} (should be close to 3.0)")
print(f"Final b: {b[0][0]:.4f} (should be close to 2.0)")
print("Training Complete!")

# 5. Plot loss curve
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Loss Decreasing Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label="Actual Data")
plt.plot(X, np.dot(X, W) + b, color='red', 
         linewidth=2, label="Model Prediction")
plt.title("Model Fitting the Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()