import pandas as pd
import math
import numpy as np

# Load dataset
data = pd.read_csv("./playtennis.csv")
features = [feat for feat in data.columns]
features.remove("answer")

# Node class for the decision tree
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

# Calculate entropy of the dataset
def entropy(examples):
    pos = 0.0
    neg = 0.0
    for _, row in examples.iterrows():
        if row["answer"] == "yes":
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0.0:
        return 0.0
    p = pos / (pos + neg)
    n = neg / (pos + neg)
    return -(p * math.log(p, 2) + n * math.log(n, 2))

# Information gain of an attribute
def info_gain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = entropy(examples)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e
    return gain

# ID3 algorithm to build the tree
def ID3(examples, attrs):
    root = Node()
    max_gain = 0
    max_feat = ""

    for feature in attrs:
        gain = info_gain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature

    root.value = max_feat
    uniq = np.unique(examples[max_feat])

    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata["answer"])[0]
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            child = ID3(subdata, new_attrs)
            dummyNode.children.append(child)
            root.children.append(dummyNode)

    return root

# Print the decision tree
def printTree(root: Node, depth=0):
    print("\t" * depth + str(root.value), end="")
    if root.isLeaf:
        print(" ->", root.pred)
    else:
        print()
    for child in root.children:
        printTree(child, depth + 1)

# Classify a new instance using the tree
def classify(root: Node, new):
    for child in root.children:
        if child.value == new[root.value]:
            if child.isLeaf:
                return child.pred
            else:
                return classify(child.children[0], new)

# Run the program
if __name__ == "__main__":
    root = ID3(data, features)
    print("Decision Tree:")
    printTree(root)
    print("\n------------------")

    new = {"outlook": "sunny", "temperature": "hot", "humidity": "normal", "wind": "strong"}
    prediction = classify(root, new)
    print("Predicted Label for new example", new, "is:", prediction)
