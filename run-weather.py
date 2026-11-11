import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from cli import display_path, keep_dashboard_open, load_trained_model, project_path, run_args, start_dashboard
from weather import CLASS_NAMES, generate_weather_dataset


# Evaluates the model saved by train-weather.py on new data: prints the precision, recall and F1 of each class and
# saves the confusion matrix to confusion-matrix.png.

args = run_args("Evaluates the model saved by train-weather.py on new data.")
model = load_trained_model(project_path('models', 'weather.joblib'), 'train-weather.py')
model.summary()
monitor = start_dashboard(args.dashboard, {"weather classifier": model})


# Confusion matrix: the rows are the true conditions and the columns the predicted ones, so the diagonal counts the
# correct predictions and every other cell shows which conditions get confused
def plot_confusion_matrix(cm, class_names, filepath):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filepath, dpi=300)


test_features, test_labels = generate_weather_dataset(100)
predicted_labels = np.argmax(model.predict(test_features), axis=1)  # the class with the highest probability

figure_filepath = project_path('confusion-matrix.png')
plot_confusion_matrix(confusion_matrix(test_labels, predicted_labels), CLASS_NAMES, figure_filepath)

# Precision: of the samples predicted as a class, how many really are. Recall: of the samples of a class, how many
# were found. F1 combines both. Accuracy alone hides which classes get confused.
print(classification_report(test_labels, predicted_labels, target_names=CLASS_NAMES, digits=3))
print(f"Confusion matrix saved to {display_path(figure_filepath)}")
keep_dashboard_open(monitor)
