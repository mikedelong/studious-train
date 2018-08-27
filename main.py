from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

print('started')
return_X_y = True
(data, target) = load_diabetes(return_X_y=return_X_y)
print('data has %d rows and %d columns' % data.shape)

model = DecisionTreeRegressor()

test_size = 0.10
for test_size in [0.7, 0.8, 0.9]:
  scores = list()
  for random_state in range(500):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
  print('with test size %.2f we have score %.3f' % (test_size, sum(scores)/len(scores)))

print('done')
