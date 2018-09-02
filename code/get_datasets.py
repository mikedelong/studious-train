from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_wine

return_X_y = False

boston_bunch = load_boston(return_X_y=return_X_y)
boston_data = boston_bunch.data
boston_target = boston_bunch.target
boston_feature_names = boston_bunch.feature_names
boston_description = boston_bunch.DESCR

cancer_bunch = load_breast_cancer(return_X_y=return_X_y)
cancer_data = cancer_bunch['data']
cancer_target = cancer_bunch['target']
cancer_feature_names = cancer_bunch['feature_names']
cancer_description = cancer_bunch['DESCR']
diabetes_bunch = load_diabetes(return_X_y=return_X_y)
diabetes_data = diabetes_bunch['data']
diabetes_target = diabetes_bunch['target']
diabetes_feature_names = diabetes_bunch['feature_names']
diabetes_description = diabetes_bunch['DESCR']
(digits_data, digits_target) = load_digits()
(iris_data, iris_target) = load_iris()
(linnerud_data, linnerud_target) = load_linnerud()
(wine_data, wine_target) = load_wine()
