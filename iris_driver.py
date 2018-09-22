from DecisionTree import *
import pandas as pd
from sklearn import model_selection

header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']

# Iris Data
df = pd.read_csv('./datasets/iris.csv',
                 header=None,
                 names=['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class'])

lst = df.values.tolist()
# print(class_counts(lst))  # {'Iris-versicolor': 50, 'Iris-setosa': 50, 'Iris-virginica': 50}
# print(entropy(lst))  # 1.58 which is max entropy, i.e. log2(k) where k is number of categories
# print(math.log(3, 2))  # 1.58 just proving the point
# print(entropy(lst[10:65]))  # .84
# print(entropy(lst[0:2]))  # 0.0 min entropy (all the same flower classification)
t = build_tree(lst, header)
print("\n\n********** Decision Tree ****************\n")
print_tree(t)

print("\n\n********** Leaf nodes ****************\n")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

print("\n\n********** Non-leaf nodes ****************\n")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("\n\n*************Tree before pruning*******\n")
print_tree(t)
acc = computeAccuracy(test, t)
print("\nAccuracy on test = " + str(acc))

# ## TODO: You have to decide on a pruning strategy
t_pruned = prune_tree(t, [12, 13, 5, 11])

print("\n\n*************Tree after pruning*******\n")
print_tree(t_pruned)
acc = computeAccuracy(test, t)
print("\nAccuracy on test = " + str(acc))
