from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Generate the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data into training and test sets
random_state = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

# Choose the number of neighbors
for n_neighbors in range(2, 10):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the classifier on training set
    clf.fit(X_train, y_train)

    # Make the predictions on test set
    predictions = clf.predict(X_test)
    print('Test set predictions: {}'.format(predictions))

    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print('neighbors: {} Test set accuracy: {:.2f}'.format(n_neighbors, accuracy))
