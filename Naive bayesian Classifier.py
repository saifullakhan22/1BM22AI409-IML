#7

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import metrics

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes Classifier
naive_bayes_classifier = GaussianNB()

# Train the classifier on the training set
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = naive_bayes_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
