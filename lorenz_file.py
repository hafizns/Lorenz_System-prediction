import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os

def read_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            contents = file.read()
            return contents
    else:
        print(f"File not found: {file_path}")
        return None

# Replace 'file_path' with the actual path to the file you want to read
file_path = "filepath"
file_contents = read_file(file_path)

if file_contents:

    file_contents = list(map(float, file_contents.split()))

    t = list(range(len(file_contents)))

x = file_contents
Nt = 2000
Np = 2000
wa = 70
data = x
dim = 1
eig_rho = 1.2
n = 1000
a = 0.3
w_in = (np.random.rand(n, 1 + dim) - 0.5) * 1
W = np.random.rand(n, n) - 0.5
rad = max(abs(linalg.eig(W)[0]))
W = W * eig_rho / rad

X = np.zeros((1 + dim + n, Nt - wa))
Yt = data[wa + 1:Nt + 1]

# training phase
x = np.zeros((n, 1))
for i in range(Nt):
    u = data[i]
    x = (1 - a) * x + a * np.tanh(np.dot(w_in, np.vstack((1, u))) + np.dot(W, x))
    if i >= wa:
        X[:, i - wa] = np.vstack((1, u, x))[:, 0]
reg = 1e-8
#Wout = linalg.solve(np.dot(X, np.transpose(X)) + reg * np.eye(1 + dim + n), np.dot(X, np.transpose(Yt))).T
Wout = np.transpose(linalg.solve(np.dot(X, np.transpose(X)) + reg * np.eye(1 + dim + n), np.dot(X, np.transpose(Yt))))

#predicting phase
Y = np.zeros((dim, Np))
u = data[Nt]
for i in range(Np):
    x = (1 - a) * x + a * np.tanh(np.dot(w_in, np.vstack((1, u))) + np.dot(W, x))
    y = np.dot(Wout, np.vstack((1, u, x)))
    Y[:, i] = y
    u = y

# Compute MSE for the first errorLen time steps
errorLen = 500
mse = sum(np.square(data[Nt + 1:Nt + errorLen + 1] - Y[0, 0:errorLen])) / errorLen
print('MSE = ' + str(mse))

# Plot some signals
xp = range(len(Y.T))
yp = Y.T
xt = data[Nt + 1:Nt + Np + 1]

plt.plot(xp, yp, label ='predicted') # length = 2K
plt.plot(xt, label = 'trained') #length = 999
plt.legend()
plt.show()import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os

def read_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            contents = file.read()
            return contents
    else:
        print(f"File not found: {file_path}")
        return None

# Replace 'file_path' with the actual path to the file you want to read
file_path = "C:/Users/hp/Desktop/1.txt"
file_contents = read_file(file_path)

if file_contents:

    file_contents = list(map(float, file_contents.split()))

    t = list(range(len(file_contents)))

x = file_contents
Nt = 2000
Np = 2000
wa = 70
data = x
dim = 1
eig_rho = 1.2
n = 1000
a = 0.3
w_in = (np.random.rand(n, 1 + dim) - 0.5) * 1
W = np.random.rand(n, n) - 0.5
rad = max(abs(linalg.eig(W)[0]))
W = W * eig_rho / rad

X = np.zeros((1 + dim + n, Nt - wa))
Yt = data[wa + 1:Nt + 1]

# training phase
x = np.zeros((n, 1))
for i in range(Nt):
    u = data[i]
    x = (1 - a) * x + a * np.tanh(np.dot(w_in, np.vstack((1, u))) + np.dot(W, x))
    if i >= wa:
        X[:, i - wa] = np.vstack((1, u, x))[:, 0]
reg = 1e-8
#Wout = linalg.solve(np.dot(X, np.transpose(X)) + reg * np.eye(1 + dim + n), np.dot(X, np.transpose(Yt))).T
Wout = np.transpose(linalg.solve(np.dot(X, np.transpose(X)) + reg * np.eye(1 + dim + n), np.dot(X, np.transpose(Yt))))

#predicting phase
Y = np.zeros((dim, Np))
u = data[Nt]
for i in range(Np):
    x = (1 - a) * x + a * np.tanh(np.dot(w_in, np.vstack((1, u))) + np.dot(W, x))
    y = np.dot(Wout, np.vstack((1, u, x)))
    Y[:, i] = y
    u = y

# Compute MSE for the first errorLen time steps
errorLen = 500
mse = sum(np.square(data[Nt + 1:Nt + errorLen + 1] - Y[0, 0:errorLen])) / errorLen
print('MSE = ' + str(mse))

# Plot some signals
xp = range(len(Y.T))
yp = Y.T
xt = data[Nt + 1:Nt + Np + 1]

plt.plot(xp, yp, label ='predicted') # length = 2K
plt.plot(xt, label = 'trained') #length = 999
plt.legend()
plt.show()