from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB





def gaussian(X,y,component):



    best_comp = dict()
    for i in range(1,component):
        gmm = mixture.GaussianMixture(n_components = i)

        gmm.fit(X)


        pred_gmm = gmm.predict(X)




        gnb = GaussianNB()

        y_pred = gnb.fit(X, y).predict(X)


        comp = adjusted_rand_score(pred_gmm, y)
        best_comp[i]=comp
        print(f"Component {i} {comp}")
    max_value=max(best_comp, key=best_comp.get)
    print(f"Best Component: {max_value}, Score: {best_comp.get(max_value)}'")
    print("**Naive Bayes Classifier**")
    print("Number of mislabeled points out of a total %d points : %d"
              % (X.shape[0], (y != y_pred).sum()))
    print("Accuracy Score:",accuracy_score(y, y_pred))


def kernel_density(X,y,bw):


    from sklearn.neighbors import KernelDensity

    ig, ax = plt.subplots()

    colors = ["navy", "cornflowerblue"]
    kernels = ["gaussian", "tophat"]
    lw = 2

    for color, kernel in zip(colors, kernels):

                kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
                X.sort(axis=0)
                log_dens = kde.score_samples(X)

                ax.plot(
                    X,
                    np.exp(log_dens),
                    color=color,
                    lw=lw,
                    linestyle="-",
                    label="kernel = '{0}'".format(kernel),
        )

    ax.text(6, 0.38, "N={0} points".format(len(X)))

    ax.legend(loc="upper left")
    ax.plot(X, -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")

    ax.set_xlim(-4, 9)
    ax.set_ylim(-0.02, 0.4)
    plt.title(f"Bandwidth = {bw}")
    plt.show()


if __name__ == '__main__':

    X, y = datasets.load_iris(return_X_y=True)


    gaussian(X,y,6)
    kernel_density(X,y,0.6)
    kernel_density(X,y,0.7)
    kernel_density(X,y,1)
