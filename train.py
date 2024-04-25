from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from feedforward import FeedForward
from etc import preprocess_text, save_model


vectorizer = TfidfVectorizer()


# Assume emails is a list of email contents
emails = [
    "Dear user, exclusive deal just for you, don't miss out!",
    "Hi there, I saw your profile online and would like to connect.",
    "Urgent: Your account has been breached. Please update your password immediately!",
    "Are you available for a meeting tomorrow at 3 PM?",
    "Congratulations! You have won a $1000 gift card. Click here to claim now!",
    "Could you please send the updated report by EOD?",
    "Hot singles in your area waiting to meet you. Click here!",
    "Reminder: Your appointment is scheduled for 10 AM tomorrow.",
    "You've been selected for a chance to get a free iPhone. Click now.",
    "Please review the attached invoice for your recent purchase.",
    "Your free trial expires today, click!",
    "Meeting today at noon",
    "Free credit report now available! Click!!!",
    "Click here to earn money!",
    "Promotion, very cheap glass. Click now",
    "Free, Free, Free, Click!!!",
    "After acquiring a dataset, the typical steps."
] * 10

targets = [[1] if "click" in email or "Click" in email else [0] for email in emails]

processed_emails = [preprocess_text(email) for email in emails]
X = vectorizer.fit_transform(processed_emails)

network = FeedForward(verbose=True)
network.add_layer(num_inputs=X.shape[1], num_neurons=20, act_fn='relu')
network.add_layer(num_inputs=20, num_neurons=20, act_fn='relu')
network.add_layer(num_inputs=20, num_neurons=1, act_fn='sigmoid')

network.train(X.toarray(), targets, epochs=100, learning_rate=0.001)

# Savel fitted model
save_model('simple_spam_classifier', network.layers)

# Save the fitted vectorizer
joblib.dump(vectorizer, 'model_vectorizer.joblib')

