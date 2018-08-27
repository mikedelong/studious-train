from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

print('started')
return_X_y = True
(data, target) = load_diabetes(return_X_y=return_X_y)
print('data has %d rows and %d columns' % data.shape)

model = DecisionTreeRegressor()

test_size = 0.10
scores = list()
for random_state in range(1000):
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
  model.fit(X_train, y_train)
  score = model.score(X_test, y_test)
  scores.append(score)
print(sum(scores)/len(scores))

print('done')
