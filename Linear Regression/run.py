import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from batch_gradient_descent import BatchGradientDescent
from stochastic_gradient_descent import StochasticGradientDescent
from analytical_gradient_descent import AnalyticalGradientDescent

train_data = "concrete/train.csv"
test_data = "concrete/test.csv"

features = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'label']

batch_gradient_descent = BatchGradientDescent()

stochastic_gradient_descent = StochasticGradientDescent()

train_df = pd.read_csv(train_data, names=features).astype(float)
test_df = pd.read_csv(test_data, names=features).astype(float)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

bgd_training_cost = batch_gradient_descent.fit(X_train, y_train)
batch_gradient_descent_error = batch_gradient_descent.get_error(X_test, y_test)
print('BatchGradientDescent weights: ' + str(batch_gradient_descent.weights))

sgd_training_cost = stochastic_gradient_descent.fit(X_train, y_train)
stochastic_gradient_descent_error = stochastic_gradient_descent.get_error(X_test, y_test)
print('StochasticGradientDescent weights: ' + str(stochastic_gradient_descent.weights))

fig1 = plt.figure(1)
ax1 = plt.axes()
ax1.plot(range(len(bgd_training_cost)), bgd_training_cost, c='b', label='Cost')
ax1.set_title("Cost rate of Batch GD")

fig2 = plt.figure(2)
ax2 = plt.axes()
ax2.plot(range(len(sgd_training_cost)), sgd_training_cost, c='b', label='Cost')
ax2.set_title("Cost rate of Stochastic GD")

# Analytical solution
analytical_gradient_descent = AnalyticalGradientDescent(X_train, y_train)
analytical_gradient_descent_error = analytical_gradient_descent.get_error(X_test, y_test)
print('AGD weights: ' + str(analytical_gradient_descent.weights))
print('AGD Error: ' + str(analytical_gradient_descent_error))

print(
    'Error in BGD weight: ' + str(np.linalg.norm(batch_gradient_descent.weights - analytical_gradient_descent.weights)))
print('Error in SGD weight: ' + str(
    np.linalg.norm(stochastic_gradient_descent.weights - analytical_gradient_descent.weights)))

plt.show()
