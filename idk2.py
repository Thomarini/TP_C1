import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix
import numpy as np

def tan_hyp_3d(x):
    A = np.array([[5, 0], [0, 7]])
    b = np.array([-0.5, -0.5])

    z = A.dot(x + b) 
    h = np.transpose(z).dot(np.ones(z.shape))
    y = (np.tanh(h) + 1) / 2 
    return y

def generate_dataset(size):
    X = np.random.rand(size, 2)
    y = np.array([tan_hyp_3d(x) for x in X])
    return X, y

def plot_3d_surface(ax, X, y, title):
    step_v = 0.05

    x1min, x1max = X[:, 0].min(), X[:, 0].max()
    x2min, x2max = X[:, 1].min(), X[:, 1].max()

    x1v = np.arange(x1min, x1max, step_v)
    x2v = np.arange(x2min, x2max, step_v)
    Xv, Yv = np.meshgrid(x1v, x2v)

    R = np.zeros(Xv.shape)
    for i, x1 in enumerate(x1v):
        for j, x2 in enumerate(x2v):
            R[i, j] = tan_hyp_3d(np.array([x1, x2]))

    surf = ax.plot_surface(Xv, Yv, R, cmap='viridis', alpha=0.5, label='True Surface')
    ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', label='Data Points')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Output')
    ax.set_title(title)
    ax.legend()

def plot_confusion(X_test, y_test, model, namefig="confusion"):
    plt.plot(y_test, model.predict(X_test), '.b')
    plt.plot(y_test, y_test, '-r')
    plt.grid()
    plt.savefig(namefig+'.jpg', dpi=200)
    plt.close()

def train_and_evaluate(X_train, y_train, X_test, y_test, hidden_layer_sizes):
    errors_train = []
    errors_test = []
    conf_matrices = []

    for i, hl_size in enumerate(hidden_layer_sizes):

        print("currently at {}%".format(i/len(hidden_layer_sizes)*100))

        # Train the model
        model = MLPRegressor(hidden_layer_sizes=(hl_size,), max_iter=2000, activation="tanh", learning_rate="adaptive")
        model.fit(X_train, y_train)

        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        mse_train = np.mean((y_train - y_train_pred)**2)
        errors_train.append(mse_train)

        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        mse_test = np.mean((y_test - y_test_pred)**2)
        errors_test.append(mse_test)

        # Confusion matrix
        """ conf_matrix = confusion_matrix((y_test > 0.5).astype(int), (y_test_pred > 0.5).astype(int))
        conf_matrices.append(conf_matrix) """
        plot_confusion(X_test, y_test, model, namefig="confusion_{}".format(hl_size))

        # Plot 3D surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_surface(ax, X_test, y_test_pred, f'Hidden Layers: {hl_size}')
        plt.savefig("{}.jpg".format(hl_size), dpi=200)
        plt.close()

    
    return errors_train, errors_test, conf_matrices

def main():
    # Generate datasets
    np.random.seed(42)
    X_train, y_train = generate_dataset(200)
    X_test, y_test = generate_dataset(1600)

    # Define hidden layer sizes to be tested
    hidden_layer_sizes = np.arange(2, 115, 1)

    # Train and evaluate
    errors_train, errors_test, conf_matrices = train_and_evaluate(X_train, y_train, X_test, y_test, hidden_layer_sizes)

    # Plot errors
    plt.figure()
    plt.plot(hidden_layer_sizes, errors_train, label='Training Error')
    plt.plot(hidden_layer_sizes, errors_test, label='Test Error')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Test Errors for Different Hidden Layer Sizes')
    plt.legend()
    plt.savefig("{}.jpg", dpi=200)

    best = np.argmin(errors_test) + 2
    model = MLPRegressor(hidden_layer_sizes=(best,), max_iter=2000, activation="tanh", learning_rate="adaptive")
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = model.predict(X_test)

    # Plot 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_surface(ax, X_test, y_test_pred, f'Hidden Layers: {best}')
    plt.show()
    
    
if __name__ == "__main__":
    main()

