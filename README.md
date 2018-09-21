# Entropy Optimized Decision Tree

Creates a decision tree to classify iris flowers from the UC Irvine Machine Learning dataset: [link](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/)

## Instructions

Use the driver to specify the dataset to which you want to apply this model.

`python3 driver.py`

## Decision Tree for Iris Data

Calculated decision tree for Iris data without pruning.

```
Is PetalW >= 1.0?   (Depth:  0 , ID:  0 )
--> True:
  Is PetalW >= 1.8?   (Depth:  1 , ID:  2 )
 --> True:
    Is PetalL >= 4.9?   (Depth:  2 , ID:  6 )
   --> True:
      Predict Iris-virginica   (Depth:  3 , ID:  14 )
   --> False:
      Is SepalW >= 3.2?   (Depth:  3 , ID:  13 )
     --> True:
        Predict Iris-versicolor   (Depth:  4 , ID:  28 )
     --> False:
        Predict Iris-virginica   (Depth:  4 , ID:  27 )
 --> False:
    Is PetalL >= 5.0?   (Depth:  2 , ID:  5 )
   --> True:
      Is PetalW >= 1.6?   (Depth:  3 , ID:  12 )
     --> True:
        Is PetalL >= 5.8?   (Depth:  4 , ID:  26 )
       --> True:
          Predict Iris-virginica   (Depth:  5 , ID:  54 )
       --> False:
          Predict Iris-versicolor   (Depth:  5 , ID:  53 )
     --> False:
        Predict Iris-virginica   (Depth:  4 , ID:  25 )
   --> False:
      Is PetalW >= 1.7?   (Depth:  3 , ID:  11 )
     --> True:
        Predict Iris-virginica   (Depth:  4 , ID:  24 )
     --> False:
        Predict Iris-versicolor   (Depth:  4 , ID:  23 )
--> False:
  Predict Iris-setosa   (Depth:  1 , ID:  1 )
```


## Pruning Strategy - Iris Data

I tried multiple pruning strategies on the Iris data. I used a test partition size of 20% for all tests. I ran each scenario 10+ times. I did not find a strategy that increased the accuracy rate without also possibly decreasing the accuracy rate. It seems that, at least for the Iris data, the tree should be left unpruned.

Here are some of the observed results:

#### Original Pruning Strategy -- 26, 11, 5

*Accuracy Before Pruning*: 90 - 100%
*Accuracy After Pruning*: 87 - 100%
*Max Accuracy Increase*: 3%
*Max Accuracy Decrease*: 6%


#### Above Leaf Nodes -- 11, 26, 13, 6

*Accuracy Before Pruning*: 90 - 100%
*Accuracy After Pruning*: 87 - 100%
*Max Accuracy Increase*: 4%
*Max Accuracy Decrease*: 7%

#### Higher Up -- 12, 13, 5, 11

*Accuracy Before Pruning*: 87 - 100%
*Accuracy After Pruning*: 87 - 100%
*Max Accuracy Increase*: %
*Max Accuracy Decrease*: 10%
