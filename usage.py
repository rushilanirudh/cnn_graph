import tensorflow as tf
from lib import models, graph, coarsening, utils
import numpy as np
import matplotlib.pyplot as plt


d = 100    # Dimensionality.
n = 10000  # Number of samples.
c = 5      # Number of feature communities.

# Data matrix, structured in communities (feature-wise).
X = np.random.normal(0, 1, (n, d)).astype(np.float32)
X += np.linspace(0, 1, c).repeat(d // c)

# Noisy non-linear target.
w = np.random.normal(0, .02, d)
t = X.dot(w) + np.random.normal(0, .001, n)
t = np.tanh(t)
plt.figure(figsize=(15, 5))
plt.plot(t, '.')

# Classification.
y = np.ones(t.shape, dtype=np.uint8)
y[t > t.mean() + 0.4 * t.std()] = 0
y[t < t.mean() - 0.4 * t.std()] = 2
print(y.shape)
print('Class imbalance: ', np.unique(y, return_counts=True)[1])


# Then split this dataset into training, validation and testing sets.

# In[6]:

n_train = n // 2
n_val = n // 10

X_train = X[:n_train]
X_val   = X[n_train:n_train+n_val]
X_test  = X[n_train+n_val:]

y_train = y[:n_train]
y_val   = y[n_train:n_train+n_val]
y_test  = y[n_train+n_val:]


dist, idx = graph.distance_scipy_spatial(X_train.T, k=10, metric='euclidean')
A = graph.adjacency(dist, idx).astype(np.float32)

assert A.shape == (d, d)
print('d = |V| = {}, k|V| < |E| = {}'.format(d, A.nnz))
plt.spy(A, markersize=2, color='black');



graphs, perm = coarsening.coarsen(A, levels=3, self_connections=False)

X_train = coarsening.perm_data(X_train, perm)
X_val = coarsening.perm_data(X_val, perm)
X_test = coarsening.perm_data(X_test, perm)

# In[9]:

L = [graph.laplacian(A, normalized=True) for A in graphs]
graph.plot_spectrum(L)



# In[10]:

params = dict()
params['dir_name']       = 'demo'
params['num_epochs']     = 40
params['batch_size']     = 100
params['eval_frequency'] = 200

# Building blocks.
params['filter']         = 'chebyshev5'
params['brelu']          = 'b1relu'
params['pool']           = 'apool1'

# Number of classes.
C = y.max() + 1
assert C == np.unique(y).size

# Architecture.
params['F']              = [32, 64]  # Number of graph convolutional filters.
params['K']              = [20, 20]  # Polynomial orders.
params['p']              = [4, 2]    # Pooling sizes.
params['M']              = [512, C]  # Output dimensionality of fully connected layers.

# Optimization.
params['regularization'] = 5e-4
params['dropout']        = 1
params['learning_rate']  = 1e-3
params['decay_rate']     = 0.95
params['momentum']       = 0.9
params['decay_steps']    = n_train / params['batch_size']


# In[11]:

model = models.cgcnn(L, **params)
accuracy, loss, t_step = model.fit(X_train, y_train, X_val, y_val)


fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.plot(accuracy, 'b.-')
ax1.set_ylabel('validation accuracy', color='b')
ax2 = ax1.twinx()
ax2.plot(loss, 'g.-')
ax2.set_ylabel('training loss', color='g')
plt.show()


# In[ ]:

print('Time per step: {:.2f} ms'.format(t_step*1000))


# In[ ]:

res = model.evaluate(X_test, y_test)
print(res[0])
