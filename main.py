from deeptorch.neuron import *

def main(X,y):
    neuron = Neuron(X.shape[1])
    neuron.fit(X,y,1000,1e-5,False,frequency=100)
    y_pred = neuron.predict(X)
    print(f"Accuracy : {round(accuracy(y,y_pred),4)} on {X.shape[0]} samples")

if __name__ == "__main__":
    X_r = np.random.randn(1000,6)
    y_r = np.array([0,1]*500)
    np.random.shuffle(y_r)

    X = np.random.randn(10000,8)
    y = np.array((5*X**2+23*X+np.random.sample((X.shape))).sum(axis=1) > 38,dtype=float) 

    main(X,y)
