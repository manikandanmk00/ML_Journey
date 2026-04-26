import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

# Network Architecture:
# Input(2) → Hidden Layer(4 neurons) → Output(1)
#
# shape?
# 2 input features
# 4 hidden neurons 
# 1 output 

# Create non-linear data
# Linear regression CANNOT fit this!
X = np.random.randn(200, 2)  # 200 samples, 2 features
y = ((X[:, 0]**2 + X[:, 1]**2) > 1).astype(float)
y = y.reshape(-1, 1)

print("X shape:", X.shape)  # (200, 2)
print("y shape:", y.shape)  # (200, 1)
print("Sample X:", X[:3])
print("Sample y:", y[:3])

# Visualize our data
plt.scatter(X[:, 0], X[:, 1], 
            c=y.flatten(), 
            cmap='RdYlBu', 
            alpha=0.7)
plt.title("Our Non-Linear Data\n(Linear Regression CANNOT fit this!)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Class")
plt.savefig("data.png")
plt.show()
print("Data is circular - impossible for linear regression!")



# Network: Input(2) → Hidden(4) → Output(1)

# Layer 1 weights: 2 inputs → 4 hidden neurons
W1 = np.random.randn(2, 4) * 0.01  # Small random init
b1 = np.zeros((1, 4))

# Layer 2 weights: 4 hidden → 1 output  
W2 = np.random.randn(4, 1) * 0.01
b2 = np.zeros((1, 1))

print("W1 shape:", W1.shape)  # (2, 4)
print("b1 shape:", b1.shape)  # (1, 4)
print("W2 shape:", W2.shape)  # (4, 1)
print("b2 shape:", b2.shape)  # (1, 1)
print("Network initialized!")



def relu(x):
    """
    ReLU activation
    Negative → 0, Positive → stays
    Used in hidden layers
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU for backpropagation
    Your calculus! 
    If x > 0: derivative = 1
    If x < 0: derivative = 0
    """
    return (x > 0).astype(float)

def sigmoid(x):
    """
    Sigmoid activation
    Squashes output between 0 and 1
    Perfect for binary classification (our task!)
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid for backpropagation
    s'(x) = s(x) * (1 - s(x))
    """
    s = sigmoid(x)
    return s * (1 - s)

# Test your activation functions
test = np.array([-3, -1, 0, 1, 3])
print("Input:          ", test)
print("ReLU output:    ", relu(test))
print("Sigmoid output: ", sigmoid(test).round(3))


def forward_pass(X, W1, b1, W2, b2):
    """
    Full forward pass through neural network
    
    Layer 1: Linear transformation
    Layer 1: ReLU activation (non-linearity!)
    Layer 2: Linear transformation  
    Layer 2: Sigmoid activation (probability output)
    """
    # Layer 1
    Z1 = np.dot(X, W1) + b1    # Linear: (200,2)*(2,4) = (200,4)
    A1 = relu(Z1)               # Activation: (200,4)
    
    # Layer 2
    Z2 = np.dot(A1, W2) + b2   # Linear: (200,4)*(4,1) = (200,1)
    A2 = sigmoid(Z2)            # Activation: (200,1) → probabilities!
    
    # Store for backprop
    cache = {
        'Z1': Z1, 'A1': A1,
        'Z2': Z2, 'A2': A2
    }
    
    return A2, cache

# Test forward pass
output, cache = forward_pass(X, W1, b1, W2, b2)
print("Output shape:", output.shape)      # (200, 1)
print("Output range:", output.min().round(4), 
      "to", output.max().round(4))        # 0 to 1
print("These are PROBABILITIES of being class 1!")



def compute_loss(y_actual, y_predicted):
    """
    Binary Cross Entropy Loss
    Better than MSE for classification!
    
    Loss = -1/n * sum(y*log(pred) + (1-y)*log(1-pred))
    
    You know logarithms from mathematics!
    When prediction is perfect: loss = 0
    When prediction is wrong: loss = high
    """
    n = len(y_actual)
    
    # Clip to avoid log(0) = infinity
    y_predicted = np.clip(y_predicted, 1e-7, 1-1e-7)
    
    loss = -1/n * np.sum(
        y_actual * np.log(y_predicted) + 
        (1 - y_actual) * np.log(1 - y_predicted)
    )
    return loss

initial_output, cache = forward_pass(X, W1, b1, W2, b2)
initial_loss = compute_loss(y, initial_output)
print(f"Initial loss: {initial_loss:.4f}")




def backward_pass(X, y, cache, W1, W2):
    """
    Backpropagation = Chain Rule from calculus
    
    You learned chain rule in university:
    d/dx[f(g(x))] = f'(g(x)) * g'(x)
    
    We apply this backwards through the network
    to find gradients for ALL weights
    """
    n = len(y)
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']
    
    # Output layer gradients
    dZ2 = A2 - y                              # (200,1)
    dW2 = (1/n) * np.dot(A1.T, dZ2)          # (4,1)
    db2 = (1/n) * np.sum(dZ2, axis=0)        # (1,1)
    
    # Hidden layer gradients (chain rule!)
    dA1 = np.dot(dZ2, W2.T)                  # (200,4)
    dZ1 = dA1 * relu_derivative(Z1)          # (200,4)
    dW1 = (1/n) * np.dot(X.T, dZ1)          # (2,4)
    db1 = (1/n) * np.sum(dZ1, axis=0)       # (1,4)
    
    gradients = {
        'dW1': dW1, 'db1': db1,
        'dW2': dW2, 'db2': db2
    }
    
    return gradients

gradients = backward_pass(X, y, cache, W1, W2)
print("Gradients computed successfully!")
print("dW1 shape:", gradients['dW1'].shape)
print("dW2 shape:", gradients['dW2'].shape)




# ============================================
# FULL NEURAL NETWORK TRAINING FROM SCRATCH
# ============================================

# Reset weights
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.01
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.01
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 2000
loss_history = []
accuracy_history = []

print("Training Neural Network...")
print("=" * 55)

for epoch in range(epochs):
    
    # Forward pass
    A2, cache = forward_pass(X, W1, b1, W2, b2)
    
    # Compute loss
    loss = compute_loss(y, A2)
    loss_history.append(loss)
    
    # Compute accuracy
    predictions = (A2 > 0.5).astype(float)
    accuracy = np.mean(predictions == y) * 100
    accuracy_history.append(accuracy)
    
    # Backward pass
    grads = backward_pass(X, y, cache, W1, W2)
    
    # Update weights
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    
    # Print progress
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | "
              f"Loss: {loss:.4f} | "
              f"Accuracy: {accuracy:.1f}%")

print("=" * 55)
print(f"Final Loss: {loss:.4f}")
print(f"Final Accuracy: {accuracy:.1f}%")
print("Training Complete! 🎉")

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Loss curve
axes[0].plot(loss_history, color='red')
axes[0].set_title("Loss Over Time")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

# Plot 2: Accuracy curve
axes[1].plot(accuracy_history, color='green')
axes[1].set_title("Accuracy Over Time")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy %")
axes[1].grid(True)

# Plot 3: Decision boundary
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 100),
    np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z, _ = forward_pass(grid, W1, b1, W2, b2)
Z = Z.reshape(xx.shape)

axes[2].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
axes[2].scatter(X[:,0], X[:,1], 
                c=y.flatten(), 
                cmap='RdYlBu', 
                alpha=0.7)
axes[2].set_title("Decision Boundary\n(Neural Network learned the circle!)")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("neural_network_results.png")
plt.show()