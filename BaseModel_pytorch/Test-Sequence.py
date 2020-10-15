from collections.abc import Sequence


class Container(Sequence):

    def __init__(self, L):

        super().__init__()
        self.L = L

    def __getitem__(self, i):

        return self.L[i]

    def __len__(self):

        return len(self.L)


# Let's test it:
myobject = MyClass([1, 2, 3])

print(myobject[1:3])

#try:

#    for idx, _ in enumerate(myobject):
#        print(myobject[idx])

#except Exception:

#    print("Gah! No good!")
#    raise