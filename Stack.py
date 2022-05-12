from ProductionList import ProductionList

class Stack(ProductionList):
    def __init__(self):
        self.List = ProductionList('')

    def add(self, Object):
        # Problem 2. complete the add function of Stack
        # remember Stack has LIFO characteristics
        self.List.addLast(Object)

    def get(self):
        # Problem 2. complete the remove function of Stack
        # remember Stack has LIFO characteristics
        node = self.List.removeLast()
        return node

    def getSize(self):
        size = self.List.getSize()
        return size

    def getListString(self):
        string = self.List.getListString()
        return string