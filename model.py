import numpy as np
import struct
import os

class Network :
  def __init__(self) :
    self.layers = []
    self.layers.append(Layer(784, 256))
    self.layers[0].set_activation("RELU")
    self.layers.append(Layer(256, 64))
    self.layers[1].set_activation("RELU")
    self.layers.append(Layer(64, 10))
    self.layers[2].set_activation("SOFTMAX")

  def add_layer(self, layer) :
    self.layers.append(layer)
    
  def forward(self, input) :
    self.input = input
    for layer in self.layers :
      input = layer.forward(input)
    return input
  
  def backward(self, one_hot) :
    for layer in reversed(self.layers) :
      one_hot = layer.calculate_gradients(one_hot)
  
  def update_parameters(self, learning_rate, batch_size) :
    for layer in self.layers :
      layer.update_parameters(learning_rate, batch_size)
    
  def train(self, epochs, batch_size, learning_rate) :
    train_images, train_labels = read_mnist(dataset="training")
    iterations = int(len(train_labels) / batch_size)
    for i in range(0, epochs) :
      iteration_loss = 0
      for j in range(0, iterations) :
        for layer in self.layers:
          layer.dldw = np.zeros_like(layer.weights)
          layer.dldb = np.zeros_like(layer.biases)
        for k in range(0, batch_size) :
          index = j * batch_size + k
          prediction = self.forward(train_images[index])
          one_hot = np.zeros(10)
          one_hot[train_labels[index]] = 1
          self.backward(one_hot)
          iteration_loss += np.sum(np.square(prediction - one_hot))
        self.update_parameters(learning_rate, batch_size)
        
        if (j + 1) % 100 == 0 :
          iteration_loss /= batch_size
          print("Epoch ", i + 1, ", Iteration ", j + 1, " finished. Loss : ", iteration_loss)
          iteration_loss = 0
          learning_rate *= 0.99
          learning_rate = max(0.00001, learning_rate)
    permutation = np.random.permutation(len(train_labels))
    
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

  def evaluate(self, test_images, test_labels):
    correct_predictions = 0
    total_predictions = len(test_labels)

    for i in range(total_predictions):
      prediction = self.forward(test_images[i])
      predicted_label = np.argmax(prediction)
      if predicted_label == test_labels[i]:
          correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Test accuracy: {accuracy:.4f}")

class Layer:
  def __init__(self, input, output) :
    # He initalization
    self.weights = np.random.randn(input, output) * np.sqrt(2 / input)
    self.biases = np.random.randn(output)
    self.activation_function = "RELU"
    self.dldw = np.zeros((input, output))
    self.dldb = np.zeros(output)
  
  def set_activation(self, new_activation) :
    self.activation_function = new_activation
  
  def forward(self, input) :
    self.input = input
    self.output = np.dot(input, self.weights) + self.biases
    if self.activation_function == "RELU" :
      self.output = np.maximum(self.output, 0)
    elif self.activation_function == "SOFTMAX" :
      # self.output = np.exp(self.output) / np.sum(np.exp(self.output))
      exp_output = np.exp(self.output - np.max(self.output))
      self.output = exp_output / np.sum(exp_output)
      
    return self.output
  
  def calculate_gradients(self, gradient) :
    dldz = gradient
    if self.activation_function == "RELU" :
      dldz[self.output <= 0] = 0
    elif self.activation_function == "SOFTMAX" :
      dldz = self.output - gradient #assuming gradient is the one-hot vector
    
    dldx = np.dot(dldz, self.weights.T)
    
    self.dldw += np.dot(self.input.T.reshape(-1, 1) , dldz.reshape(1, -1))
    
    self.dldb += dldz
    
    return dldx
    
  def update_parameters(self, learning_rate, batch_size) :
    self.weights -= learning_rate * (self.dldw / batch_size)
    self.biases -= learning_rate * (self.dldb / batch_size)
    
def read_mnist(dataset="training", path=""):
    if dataset not in ["training", "testing"]:
        raise ValueError("dataset must be 'training' or 'testing'")

    if dataset == "training":
        images_file = os.path.join(path, "train-images.idx3-ubyte")
        labels_file = os.path.join(path, "train-labels.idx1-ubyte")
    else:
        images_file = os.path.join(path, "t10k-images.idx3-ubyte")
        labels_file = os.path.join(path, "t10k-labels.idx1-ubyte")

    try:
        with open(images_file, 'rb') as img_f:
            magic, num_images, rows, cols = struct.unpack(">IIII", img_f.read(16))
            images = np.frombuffer(img_f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows * cols)
            images = images/255.0
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {images_file}")

    try:
        with open(labels_file, 'rb') as lbl_f:
            magic, num_labels = struct.unpack(">II", lbl_f.read(8))
            labels = np.frombuffer(lbl_f.read(), dtype=np.uint8)
    except FileNotFoundError:
        raise FileNotFoundError(f"Label file not found: {labels_file}")

    return images, labels

if __name__ == "__main__":
    network = Network()
    epochs = 10
    batch_size = 30
    learning_rate = 0.5

    network.train(epochs, batch_size, learning_rate)

    test_images, test_labels = read_mnist(dataset="testing")
    network.evaluate(test_images, test_labels)