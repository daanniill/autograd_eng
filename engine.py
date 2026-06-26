class Value:

    # data: float, _children: tuple of two value objects, _op: operation performed on children
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
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