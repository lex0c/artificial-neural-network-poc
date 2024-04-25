import joblib

from feedforward import FeedForward
from etc import load_model, preprocess_text


network = FeedForward(layers=load_model('simple_spam_classifier'), verbose=True)


new_emails = [
    #"Win a brand new car. Click now!",
    #"Your free trial expires today, now!",
    "Your free trial expires today, click now!",
    #"Please submit the report by tomorrow."
]

vectorizer = joblib.load('model_vectorizer.joblib')

processed_new_emails = [preprocess_text(email) for email in new_emails]
X_new = vectorizer.transform(processed_new_emails)


# Using the forward method of the network to predict
predictions = network.forward(X_new.toarray()[0])
print(predictions)

