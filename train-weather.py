from cli import display_path, project_path, training_args
from data import set_seed, to_categorical
from feedforward import FeedForward
from weather import CLASS_NAMES, generate_weather_dataset


# Classification: predicts the weather condition (Sunny, Cloudy, Rainy or Snowy) from the temperature, humidity and
# wind speed, and saves the model to models/weather.joblib.

args = training_args("Classification: predicts the weather condition from the temperature, humidity and wind speed.", epochs=100)
set_seed(args.seed)

# categorical_crossentropy compares one-hot labels with the probabilities of the softmax output
features, labels = generate_weather_dataset(100)
labels_one_hot = to_categorical(labels, num_classes=len(CLASS_NAMES))

# Validation: checked after each epoch to see whether the model generalizes (it isn't used for training)
val_features, val_labels = generate_weather_dataset(20)
val_labels_one_hot = to_categorical(val_labels, num_classes=len(CLASS_NAMES))

# Test: evaluated once, at the end, on data the training never saw
test_features, test_labels = generate_weather_dataset(20)
test_labels_one_hot = to_categorical(test_labels, num_classes=len(CLASS_NAMES))


model = FeedForward()
model.add_layer(num_inputs=features.shape[1], num_neurons=16, act_fn='relu')
model.add_layer(num_inputs=16, num_neurons=len(CLASS_NAMES), act_fn='softmax')  # one probability per class
model.configure(loss='categorical_crossentropy')
model.fit_normalization(features)  # the features have different scales (humidity goes up to 100, wind up to 25)
model.summary()

# Inputs scaled to [0, 1] give small gradients, so the learning rate can be larger. log_every=10: a line every 10 epochs
model.train(features, labels_one_hot, epochs=args.epochs, learning_rate=0.1, verbose=True,
            validation_data=(val_features, val_labels_one_hot), log_every=10)

test_loss, test_accuracy = model.evaluate(test_features, test_labels_one_hot)

print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

model_filepath = project_path('models', 'weather.joblib')
model.save(model_filepath)
print(f"Model saved to {display_path(model_filepath)}")
