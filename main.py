import numpy as np                  # For linear algebra
import csv                          # For reading in data
import matplotlib.pyplot as plt     # For plotting results


def compute_cost(X, y, theta):
    cost = sum(np.power(theta[0] + theta[1] * X - y, 2))/(2*len(X))
    return cost


# # Sample data for testing algorithm (should return y_int = 0.7, slope = 0.7)
# X = [1, 2, 3, 4, 5]
# y = [1, 3, 3, 2, 5]

# Read in data
# Cost, beds, baths, sq ft
raw_data = []
# with open('houseTest.csv', newline='') as csvfile:    # Only first three houses (for testing)
with open('house.csv', newline='') as csvfile:
    my_reader = csv.reader(csvfile)
    for row in my_reader:
        raw_data.append(row)
# Delete header row
del raw_data[0]
# Convert strings to int
data = [[int(x) for x in house] for house in raw_data]

# Maybe need to normalize data: z = (x - mean) / range
X = []
y = []
for house in data:
    X.append(house[3])
    y.append(house[0])


# Need lists as arrays (vectors) for some algebra later
X = np.array(X)
y = np.array(y)

# Normalize X to be mean=0 between -1 and 1
X_mean = sum(X) / len(X)
X_range = max(X) - min(X)
X = (X - X_mean) / X_range

# Normalize y to be mean=0 between -1 and 1
y_mean = sum(y) / len(y)
y_range = max(y) - min(y)
y = (y - y_mean) / y_range

# Random starting value for y = theta0 + theta1*x
np.random.seed(0)
theta = 10000*np.random.random(2)

# Number of training sets
m = float(len(X))

# Learning rate
alpha = 0.1

# Delta theta for each step of gradient descent
dTheta = np.array([0.0, 0.0])

# Gradient descent method
epoch = 0
error = []
while epoch < 2000:
    epoch += 1
    # print(theta)
    dTheta[0] = sum(theta[0] + theta[1] * X - y) / m
    dTheta[1] = sum((theta[0] + theta[1] * X - y) * X) / m
    theta -= alpha * dTheta
    # Periodically print out cost (should be decreasing)
    if epoch % 50 == 0:
        error.append([epoch, compute_cost(X, y, theta)])
        # print(epoch, ": ", compute_cost(X, y, theta))

# Un-normalize values
X = X*X_range + X_mean
y = y*y_range + y_mean
y_int = y_range*(theta[0] - theta[1]*X_mean/X_range) + y_mean
slope = y_range*theta[1]/X_range

print("y = %f + %f x" % (y_int, slope))

plt.figure(1, figsize=(18, 8))
# Plot the linear regression with data
plt.subplot(121)    # rows, columns, which entry
plt.scatter(X, y)
plt.title('Broomfield Home Values from Zillow')
plt.xlabel('Sq. Footage')
plt.ylabel('Cost ($)')
plt.plot([0, 6000], [y_int, y_int + slope*6000], 'k-', lw=2)
plt.text(2000, 200000, "y = %f + %f x" % (y_int, slope))

# Plot the cost over iterations
plt.subplot(122)
plt.scatter(list(value[0] for value in error), list(value[1] for value in error))
plt.title('Error across Iterations')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()
