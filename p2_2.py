from matplotlib.colors import LogNorm
from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB





def gaussian(X,y,i, j, component):

    X = X[:, i:j]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    best_comp = dict()
    for i in range(1,component):
        gmm = mixture.GaussianMixture(n_components = i)

        gmm.fit(X_train)


        pred_gmm = gmm.predict(X_train)


        plt.scatter(X_train[:,0], X_train[:,1],c=pred_gmm, edgecolor='black',cmap="viridis")

        plt.title(f"Component {i}")

        gnb = GaussianNB()

        y_pred = gnb.fit(X_train, y_train).predict(X_test)

        plt.show()

        comp = adjusted_rand_score(pred_gmm, y_train)
        best_comp[i]=comp
        print(f"Component {i} {comp}")
    max_value=max(best_comp, key=best_comp.get)
    print(f"Best Component: {max_value}, Score: {best_comp.get(max_value)}'")
    print("**Naive Bayes Classifier**")
    print("Number of mislabeled points out of a total %d points : %d"
              % (X_test.shape[0], (y_test != y_pred).sum()))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))


def kernel_density(X,y,i, j,bw):
    X = X[:, i:j]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    from sklearn.neighbors import KernelDensity

    ig, ax = plt.subplots()
    # true_dens = 0.3 * norm(0, 1).pdf(X_train[:,1]) + 0.7 * norm(5, 1).pdf(X_train[:,0])
    # ax.fill(X_train,true_dens,fc="black", alpha=0.2, label="input distribution")
    colors = ["navy", "cornflowerblue"]
    kernels = ["gaussian", "tophat"]
    lw = 2

    for color, kernel in zip(colors, kernels):

                kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X_train)
                X_train.sort(axis=0)
                log_dens = kde.score_samples(X_train)

                ax.plot(
                    X_train,
                    np.exp(log_dens),
                    color=color,
                    lw=lw,
                    linestyle="-",
                    label="kernel = '{0}'".format(kernel),
        )

    ax.text(6, 0.38, "N={0} points".format(len(X_train)))

    ax.legend(loc="upper left")
    ax.plot(X_train, -0.005 - 0.01 * np.random.random(X_train.shape[0]), "+k")

    ax.set_xlim(-4, 9)
    ax.set_ylim(-0.02, 0.4)
    plt.title(f"Bandwidth = {bw}")
    plt.show()


if __name__ == '__main__':
    X, y = datasets.load_iris(return_X_y=True)
    first_feature=1
    second_feature=3
    gaussian(X,y,1,3,6)
    kernel_density(X,y,first_feature,second_feature,0.5)
    kernel_density(X,y,first_feature,second_feature,0.7)
    kernel_density(X,y,first_feature,second_feature,1)


