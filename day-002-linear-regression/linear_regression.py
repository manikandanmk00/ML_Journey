import numpy as np
import matplotlib.pyplot as plt


# Set seed for reproducibility

np.random.seed(42)


# create fake datas
# imagine: Xhours studied, y = exam score

X = np.random.randn(100,1)  # 100 students(rows), 1 feature(column)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.5

# True relationship is y = 3x + 2
# Our model needs to LEARN this from data!

print("X shape:", X.shape) #(100, 1)
print("y shape:", y.shape) #(100, 1)


# Visualize
plt.scatter(X, y, alpha=0.5)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Our training Data")
plt.savefig("data.png")
plt.show()
print("Data created successfully")