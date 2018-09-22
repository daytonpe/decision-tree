from DecisionTree import *
import pandas as pd
from sklearn import model_selection

# I'm leaving out much of the printing, as this dataset is much bigger and more complicated.

header = ['fixed acidity',
          'volatile acidity',
          'citric acid',
          'residual sugar',
          'chlorides',
          'free sulfur dioxide',
          'total sulfur dioxide',
          'density',
          'pH',
          'sulphates',
          'alcohol',
          'quality']

# Iris Data
df = pd.read_csv('./datasets/wine.csv',
                 header=None,
                 names=[
                        'fAcid',
                        'vAcid',
                        'cAcid',
                        'rSugar',
                        'chlorides',
                        'fS02',
                        'tS02',
                        'density',
                        'pH',
                        'sulfates',
                        'alcohol',
                        'quality'])

lst = df.values.tolist()

print('\n\n*****************WINE*****************')

t = build_tree(lst, header)

print("\n\n********** Decision Tree ****************\n")
print_tree(t)

# print("\n\n********** Leaf nodes ****************\n")
# leaves = getLeafNodes(t)
# for leaf in leaves:
#     print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
#
# print("\n\n********** Non-leaf nodes ****************\n")
# innerNodes = getInnerNodes(t)
# for inner in innerNodes:
#     print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
acc = computeAccuracy(test, t)
print("\nAccuracy before Pruning = " + str(acc))


# Prune one above the leaf nodes
leafParentNodes = getLeafParents(t)
print(len(leafParentNodes))
t_pruned = prune_tree(t, leafParentNodes)

acc = computeAccuracy(test, t)
print("\nAccuracy after Pruning = " + str(acc))
