class Value:

    def __init__(self, data):
        self.data = data

    # Creates string representation of class
    def __repr__(self):
        return f"Value(data={self.data})"

    # defining addition of Value objects
    def __add__(self, other):
        out = Value(self.data + other.data)
        return out
