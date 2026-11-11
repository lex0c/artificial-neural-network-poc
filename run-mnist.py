import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from cli import display_path, load_trained_model, project_path
from mnist import load_mnist


# Evaluates the model saved by train-mnist.py on the 10,000 test images: prints the accuracy, the precision, recall and
# F1 of each digit and the confusion matrix, and saves the first mistakes to mnist-errors.png.

model = load_trained_model(project_path('models', 'mnist.joblib'), 'train-mnist.py')
model.summary()

_, (test_images, test_labels) = load_mnist()

probabilities = model.predict(test_images)
predicted_labels = np.argmax(probabilities, axis=1)  # the digit with the highest probability

print(f"Test Accuracy: {np.mean(predicted_labels == test_labels)*100:.2f}%\n")
print(classification_report(test_labels, predicted_labels, digits=3))

# Rows are the true digits and columns the predicted ones: every number off the diagonal is a kind of mistake
print("Confusion matrix (rows: true digit, columns: predicted digit):")
print(confusion_matrix(test_labels, predicted_labels))

# The first 20 mistakes, each with the predicted digit, the probability the model gave it, and the true digit
mistakes = np.flatnonzero(predicted_labels != test_labels)[:20]
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 9))

for ax in axs.ravel():
    ax.axis('off')

for ax, i in zip(axs.ravel(), mistakes):
    ax.imshow(test_images[i].reshape(28, 28), cmap='gray_r')
    ax.set_title(f"predicted {predicted_labels[i]} ({probabilities[i, predicted_labels[i]]:.0%})\ntrue {test_labels[i]}", fontsize=10)

plt.tight_layout()
figure_filepath = project_path('mnist-errors.png')
plt.savefig(figure_filepath, dpi=150)
print(f"\nThe first mistakes saved to {display_path(figure_filepath)}")
