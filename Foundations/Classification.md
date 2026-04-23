# What is Classification?
Is the task of learning a mapping function f(X) --> Y
 - Given input vector X of features, predict Y
 - Where Y is a discrete, finite set of categories,
 - Optimising over some loss over a training distribution
 - With the goal of generalising to unseen samples from the same distribution

## Types of classification
### Binary: Two possible outcomes
### Multi-class: More than two classes
#### LogReg, SVM, binary classifiers
can train multiple classifiers using meta strategies: 

##### OVR: One vs Rest
 - One binary classifier for each class, classifying target class vs "rest" (all other classes)
 - Macro: Calculated the metric for each class, then take unweighted mean, doesnt accout for class distribution
 - Micro: Aggregate the TP/FP/TN/FN counts then compute metrics globally
 - Weighted Average: Calculate metric for each class, then average them but WEIGHTED by the support (the number of true instances for each label TP+FN)

##### OVO: One vs one
 - One binary classifier for each unique class pair eg. classes A B C
    - A vs B: 1 (A)
    - B vs C: 0 (C)
    - C vs A: 0 (A)
 - To get the metric, 
    1) Voting, predicted class determined by majority (A:2, C:1 = A)
    2) Ties can be broken if classifier supports confidience scores (predict_proba)
 - Num classifiers scale faster generally more than OVR
    - but each only trained on 2 classes: smaller more balanced subsets

#### LogReg and neural networks
Instead of sigmoid activation, we have **multinomial LogReg**
 - Binary LogReg: One output --> sigmoid --> probability of the positive class
 - Multiclass Logreg: One output per class --> softmax --> probability distribution over all classes

##### NNs: One output neuron per class in final layer
When softmax applied fo final layer, we get a probablity distribution summing to 1
 - prediction will be the argmax
Loss function changes from binary cross-entropy for binary classification
 - to categorical cross entropy 

### Tree-based algorithms
Decision trees can handle multiple classes innately
#### (Simple Example)
 - Petal length vs petal width scatter plot
 - length < 2.5 = x
 - lendght >= 2.5
    - width <  1.8 = △
    - width >= 1.8 = □
            
### Multi-label: An instance can belong to multiple classes simulataneously

# The Loss Function
## Cross-Entropy Loss 
Penalises the model based on how far the predicted probability is from the actual label
 - If label = 1
 - Predict 0.99 loss = tiny
 - Predict 0.01 loss = huge near infinity

$ \mathcal{L} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c}) $


# Evaluating Classifiers
## Accuracy: (TP + TN) / (TP + TN + FP + FN)
## Precision, Recall, False Discovery Rate, False Positive Rate
### Precision: TP / (TP + FP)
### Recall: TP / (TP + FN)
### Specificity: TN / (TN + FP)
### False Omisson Rate: FN / (FN + TN)
### False Discovery Rate: FP / (FP + TP)
### False Positive Rate: FP / (FP + TN)
## F1
## Weighted F1
## Matthews Correlation Coefficient (MCC)
Only produces a high score if ta model performed well in all four quadrants of the confusion matric

$ (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) $
## Cohen's Kappa
Compares observed accuracy against expected accuracy (if model just guessed based on class distribution)
 - < 0 = Worse than chance
 - 0.61 - 0.8: Substantial agreement
 - 0.81 - 1.00: Almost perfect agreement



# Types of Classifiers
## Decision Tree
## Random Forest, Boosting, Bagging
## Logistic Regression/Regression-styled classifiers?
## Neural Networks

# Training Classifiers
## Train/test/val split
## K-fold cross validation



