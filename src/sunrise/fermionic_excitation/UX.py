import tequila as tq
import sunrise as sun
import numpy as np


class A:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        raise TypeError(f"Class A cannot be directly added to {type(other).__name__}")

    def __repr__(self):
        return f"A({self.value})"

class B:
    def __init__(self, value):
        self.value = value

    def __radd__(self, other):
        print(f"__radd__ of B called with other: {other}")
        if isinstance(other, A):
            return A(other.value + self.value)
        return NotImplemented  # Let the other object's __add__ try

    def __repr__(self):
        return f"B({self.value})"

# Example usage
a_instance = A(5)
b_instance = B(3)

result = a_instance + b_instance
print(f"Result of A + B: {result}")

result_reversed = b_instance + a_instance
print(f"Result of B + A: {result_reversed}")