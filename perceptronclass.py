import numpy as np
class SigmoidPerceptron():

  def __init__(self, input_size):
    self.weights = np.random.randn(input_size)
    self.bias = np.random.randn(1)


  def sigmoid(self, z):
    return 1/(1+ np.exp(-z))


  def predict(self, inputs):
    weighted_sum = np.dot(inputs, self.weights) + self.bias
    return self.sigmoid(weighted_sum)


# inputs → all training data (features)
# targets → correct answers (0 or 1)
# learning_rate → how big each correction step is
# num_epochs → how many times the model sees the full dataset

  def fit(self, inputs, targets, learning_rate, num_epochs):
    #Counts how many data points exist
    num_examples = inputs.shape[0]

    for epoch in range(num_epochs): #Model sees entire dataset multiple times

      for i in range(num_examples):  #Go through data one row at a time

        input_vector = inputs[i] #Select one data point

        target = targets[i] #Get correct label for this input 0 or 1

        prediction = self.predict(input_vector) #Model guesses output

        error = target - prediction

        # update weights, Move weights in direction that reduces error
        gradient_weights = error * prediction * (1-prediction) * input_vector
        self.weights += learning_rate * gradient_weights

        # update bias, Adjust bias separately
        gradient_bias = error * prediction * (1-prediction)
        self.bias += learning_rate * gradient_bias


  #This checks how well the model learned.
  def evaluate(self, inputs, targets):

    correct = 0 #Tracks how many predictions are right.

    for input_vector, target in zip(inputs, targets):
      prediction = self.predict(input_vector)

      if prediction >= 0.5:
        predicted_class = 1

      else:
        predicted_class = 0

      #Increase count if prediction is right
      if predicted_class == target:
        correct += 1

    accuracy = correct / len(inputs) #accuracy = no. of correct predictions / total no. of data points
    return accuracy


#Initialize weights and bias

# Predict using weighted sum + sigmoid

# Calculate error

# Update weights and bias

# Repeat for many epochs

# Evaluate accuracy