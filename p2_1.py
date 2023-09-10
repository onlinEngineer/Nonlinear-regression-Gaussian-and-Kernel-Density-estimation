import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def linear_reg(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    linear_model_1 = linear_model.LinearRegression()

    # Training
    linear_model_1.fit(X_train, y_train)




    # Predicition by using linear reg
    pred_lin_reg = linear_model_1.predict(X_test)

    # Linear Regression Graph
    plt.title("Linear Regression Graph")
    plt.scatter(X_test, y_test)
    plt.plot(X_test, pred_lin_reg, color="red")
    plt.show()

    print("************* Linear Regression *************")
    # Linear
    print("Linear Coefficient", linear_model_1.coef_)
    print("Linear Intercept", linear_model_1.intercept_)

    # The mean squared error
    print('Linear Mean squared error: %.2f'
          % mean_squared_error(y_test, pred_lin_reg))
    # The coefficient of determination: 1 is perfect prediction
    print('Linear Coefficient of determination: %.2f'
          % r2_score(y_test, pred_lin_reg))

    print("********************************************")
    print()

def Gaussian(X,y,degree):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Gaussian Kernel Regression
    from sklearn.gaussian_process import GaussianProcessRegressor
    clf = GaussianProcessRegressor(n_restarts_optimizer=degree, random_state=42)

    # Training
    clf.fit(X_train, y_train)
    X_test.sort(axis=0)

    # Predicition by using gaussian reg
    y_pred = clf.predict(X_test)

    print(f"************* Gaussian Kernel Regression Degree: {degree} *************")
    # Gaussian

    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))

    print("********************************************")

    # Gaussian Regression Graph
    plt.title(f"Gaussian Kernel Regression Graph, Degree: {degree}")
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred, color='red')
    plt.show()



def Poly_reg(X,y,degree):


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



    poly_model = make_pipeline(PolynomialFeatures(degree=degree), linear_model.LinearRegression())
    # Sorting test data to draw just 'one line' for gaussian and polynomial
    poly_model.fit(X_train, y_train)

    X_test.sort(axis=0)

    # Predicition by using linear reg
    pred_poly_reg=poly_model.predict(X_test)
    print(f"************* Polynomial Regression Degree: {degree}*************")

    print("Polymomial Coefficient", poly_model.steps[1][1].coef_)
    print("Polymomial Intercept", poly_model.steps[1][1].intercept_)

    print('Polymomial Mean squared error: %.2f'
          % mean_squared_error(y_test, pred_poly_reg))
    # The coefficient of determination: 1 is perfect prediction
    print('Polymomial Coefficient of determination: %.2f'
          % r2_score(y_test, pred_poly_reg))

    print("********************************************")
    print()

    # Polynomial Regression Graph
    plt.title(f"Polynomial Regression Graph, Degree={degree}")
    plt.scatter(X_test,y_test)
    plt.plot(X_test,pred_poly_reg,color="red")
    plt.show()







if __name__ == '__main__':
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    diabetes_X = diabetes_X[:, np.newaxis, 4]
    linear_reg(diabetes_X,diabetes_y)
    for i in range(2,10):
        Poly_reg(diabetes_X, diabetes_y, i)
        Gaussian(diabetes_X,diabetes_y,i)

