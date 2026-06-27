import math


class Value:

    # data: float, _children: tuple of two value objects, _op: operation performed on children
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # Creates string representation of class
    def __repr__(self):
        return f"Value(data={self.data})"

    # defining addition of Value objects
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    # defining multiplication of Value objects
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    # implementing tanh as acitvation function
    # reason why we are not breaking this down into smaller functions is because we don't need to.
    # all we need to do is be able to find the local derivative, then from there we can construct a global derivative
    # since we can differentiate tanh we can just implement it here, rather than implementing exp and divide
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        return out
    