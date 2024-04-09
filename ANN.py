import numpy as np

X=np.array(([2,5],[6,3],[9,4]),dtype=float)/9
y=np.array(([92], [86], [89]),dtype=float)/100

class NN:
    def __init__(self) -> None:
        self.w1=np.random.randn(2,3)
        self.w2=np.random.randn(3,1)
    def forward(self,X):
        self.z2=1/(1+np.exp(-np.dot(X,self.w1)))
        return 1/(1+np.exp(-np.dot(self.z2,self.w2)))
    def backward(self, X, y, o):
        o_delta = (y - o) * o * (1 - o)
        self.w2 += np.dot(self.z2.T, o_delta)
        self.w1 += np.dot(X.T, np.dot(o_delta, self.w2.T) * self.z2 * (1 - self.z2))
    


NN=NN()
for _ in range(1000):
    o=NN.forward(X)
    NN.backward(X,y,o)
    print("Inputs",X,"Actual outputs ",y,"Predictons",o,"loss",np.mean(np.square(y-o)))