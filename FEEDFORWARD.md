# FeedForward

This custom neural network framework provides a basic implementation for creating and training feedforward neural networks. It includes several activation functions, a method for loss computation, and utilities for saving and loading models.

## Usage

### Create Model

Create a neural network by instantiating the `FeedForward` class and adding layers to it:

```python
from feedforward import FeedForward

model = FeedForward(verbose=True)
model.add_layer(num_inputs=3, num_neurons=5, act_fn='relu')
model.add_layer(num_inputs=5, num_neurons=1, act_fn='sigmoid')
model.summary()
```

### Train the Model

Provide training data and target values to train the network:

```python
inputs = [[-1.7021547074061765], [-2.34438654901332], [-4.749754186552664], [-5.56165352474812], [9.078507501353958]]
targets = [[-4.404309414812353], [-5.68877309802664], [-10.499508373105328], [-12.12330704949624], [17.157015002707915]]

model.train(inputs, targets, epochs=100, learning_rate=0.001)

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.2f}")
```

### Save and Load Models

Save the trained model to a file and load it later:

```python
# Saving the model
model.save('path_to_save_models/model.joblib')

# Loading the model
from feedforward import load_model

model = load_model('path_to_save_model/model.joblib')
```

### Making Predictions

Use the trained network to make predictions on new data:

```python
predictions = model.predict([6.179484095266723])
print(predictions) # ~11.35
```

### Clone the model

```python
from feedforward import clone_model

new_model = clone_model(model)
new_model.summary()

# Update the new model weights using the original model weights
new_model.set_layers(model.get_layers())
```

