import numpy as np
from keras.models import Sequential
from keras.layers import Dense

training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_data = np.array([[0], [1], [1], [0]])
test_data = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [1, 1], [1, 0], [0, 1]])

model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=600, verbose=2)

print(model.predict(test_data).round())