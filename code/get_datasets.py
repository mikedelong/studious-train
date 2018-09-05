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

digits_bunch = load_digits(return_X_y=return_X_y)
digits_data = digits_bunch['data']
digits_target = digits_bunch['target']
digits_target_names = digits_bunch['target_names']
digits_images = digits_bunch['images']
digits_description = digits_bunch['DESCR']

iris_bunch = load_iris(return_X_y=return_X_y)
iris_data = iris_bunch['data']
iris_target = iris_bunch['target']
iris_target_names = iris_bunch['target_names']
iris_feature_names = iris_bunch['feature_names']
iris_description = iris_bunch['DESCR']

linnerud_bunch = load_linnerud(return_X_y=return_X_y)
linnerud_data = linnerud_bunch['data']
linnerud_feature_names = linnerud_bunch['feature_names']
linnerud_target = linnerud_bunch['target']
linnerud_target_names = linnerud_bunch['target_names']
linnerud_description = linnerud_bunch['DESCR']
wine_bunch = load_wine(return_X_y=return_X_y)
