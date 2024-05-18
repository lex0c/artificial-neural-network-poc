# FeedForward

This custom neural network framework provides a basic implementation for creating and training feedforward neural networks. It includes several activation functions, a method for loss computation, and utilities for saving and loading models.

## Usage

### Create Model

Create a neural network by instantiating the `FeedForward` class and adding layers to it:

```python
from feedforward import FeedForward

model = FeedForward()
model.add_layer(num_inputs=3, num_neurons=5, act_fn='relu')
model.add_layer(num_inputs=5, num_neurons=1, act_fn='sigmoid')
model.configure(loss='mse')
model.summary()
```

#### Loss Functions

- **mse**: Calculates the average squared difference between the predicted values and the actual values. It penalizes larger errors more severely.
- **sparse_categorical_crossentropy**: Computes the cross-entropy loss for each sample. It’s more memory efficient than categorical crossentropy because it does not require one-hot encoding of the labels.
- **categorical_crossentropy**: Measures the cross-entropy loss between the true labels (one-hot encoded) and the predicted probabilities. It penalizes incorrect classifications by considering the predicted probability of the true class.

### Train the Model

Provide training data and target values to train the network:

```python
inputs = [[-1.7021547074061765], [-2.34438654901332], [-4.749754186552664], [-5.56165352474812], [9.078507501353958]]
targets = [[-4.404309414812353], [-5.68877309802664], [-10.499508373105328], [-12.12330704949624], [17.157015002707915]]

model.train(inputs, targets, epochs=100, learning_rate=0.001, verbose=True)

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.2f}")
```
- **inputs**: Input data for training.
- **targets**: Target labels for training.
- **epochs**: Number of epochs to train the network.
- **learning_rate**: Learning rate for weight updates.
- **batch_size**: Size of each batch (default is 32).
- **l1_lambda**: L1 regularization parameter (default is 0.0).
- **l2_lambda**: L2 regularization parameter (default is 0.0).
- **use_granular_update**: If true, uses a more granular backward, updating each weight individually (slower, default is False).
- **verbose**: If True, print detailed training progress (default is False).

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
predictions = model.predict([6.179484095266723], verbose=True)
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

