class PlanNode:
    def __init__(self, numNo, strSerialNumber, strModel):
        self.numNo = numNo
        self.strSerialNumber = strSerialNumber
        self.strModel = strModel

    def printOut(self):
        print('No :', self.numNo, ', SerialNum : ', self.strSerialNumber, ',Model:', self.strModel)

    def getNextNode(self):
        node = self.nextNode
        return node

    def getPrevNode(self):
        node = self.prevNode
        return node

    def setNextNode(self, node):
        # Problem 1. complete this method
        self.nextNode = node

    def setPrevNode(self, node):
        # Problem 1. complete this method
        self.prevNode = node