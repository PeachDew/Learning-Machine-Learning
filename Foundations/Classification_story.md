### What is the goal in machine learning?
ŷ = f(x)
 - We want to build/find a function f(x) that gives us y values, predictions, that comes close to the actual y values.

Now lets move on to the task of classification
    - classification simply means our y, the target/labels are
    - discrete: you can't be in-between two categories
    - finite: limited number of possible categories (regression infinite)
    - set of categories

### What is the simplest possible classifier?
 - ŷ = 0.5, we always predict a constant
The next simplest, we have linear regression
Here we have
 - ŷ = a . x, linear function

#### What is x and what is a
#### Simple Example: Heart disease
We have one patient, we want to predict whether they have heart disease (y = 0 (no disease) or 1 (yes disease))

So to predict something what do we need? Data
Their
x1 = age
x2 = blood pressure
x3 = cholesterol level
x4 = whether they own a soft toy

So these are raw facts, measurable measurements, so x in this context 
 - is a vector, a list of numbers, one per measurement
 - each number is one feature
 - and x is a collections of measurement for one observation/human/row

So with just x we can't really do anything,
to turn all this into a prediction y,
we need to answer:
`how much does each feature contribute?`
`is age more important as a feature than blood pressure?`
`does whether they own a soft toy even matter?`

Which part of the linear function gives us this?
 - yes, that is what a encodes,
 - it is a vector the same length as x
 - where in contains a list of weights, 
    - what does each weight mean?
    - it says how much the corresponding feature contributes to the prediction y_hat

So three letters, question
which one is for the prediction?
which one is given, unchangeable?
which one is learnable?
what does this (dot) mean in the equation
 - dot product, means we multiply corresponding entries and sum them up
 - each feature x_i get scaled by its weight a_i and the results are summed to a single number in this case y_hat

`What is ŷ here? In the context of classification what does its number mean?`
Simpler question, what type of values do we want to predict in classification?
 - a probability that it belongs to a certain class (continuous)
 - the class label (discrete)

### Before we go further into a linear regressor for classification
lets get a deeper intuition for the bias term
### Intuition for bias
[For the next step up, let me give an example]
We want to predict if a student passes 1 = pass, o = fail
given x = number of hours studied

so my task for you is to incorporate this true relationship
 - where the student have a baseline passing rate of 30%
 - with x = 0 hours studied

Can someone draw a linear function that both satisfies this equation ŷ = a . x and this true relationship?

So this true relationship, is something we'll never have in real life,
we are only working backwards as an example
because if we have convenient access to the true relationship, 
we don't need to build a ML model in the first place

without bias model/function in anchored to origin
 -> [Elaborate: which is often too severe a restriction for an already simple model in the first place
 -> it doesn't give it the flexibility to incorporate rules/patterns such as this one]

if we add a bias term, now draw a line that incorporates this relationship
 -> in a way we are giving the model more complexity and flexibility
 -> to capture more complex relationships
 -> w controls the rotation -> how steep the slope is
 -> b controls the translation -> where it sits on the origin

### Motivation for logreg
for a linear function, the output is discrete or continuous?
 - yes it is continuous by nature and
 - naturally outputs a real number along an infinite number line,

so one way we can turn continuous output to discrete class labels is with a threshold
 - <= 0.5 we get we classify as cat
 - > 0 we classify as mouse

`What is wrong with thresholding a linear output?`
`What is the range of values a linear function can output? (assuming a != 0)`
 - When we have an unbounded range, thresholding is meaningless beause....
 - what does a probability of -0.10 mean?
 - what does a probability of 23 mean?
 - Things stop making sense

### What are some ways to solve this problem?
### How can we squish the range of values our function/model outputs nicely into [0,1]? A range of values that actually can correspond to probability?
Sigmoid function --> Log reg

### Now that we have a probability, how do we convert that to an actual label?
Simplest: 0.5 threshold, >= 0.5, class label = 1, else 0

### [Q] How does the sigmoid function help linear regression do classification better?
 - original motivation, to model binary outcomes/probabilities/responses
 - p(y=1|x) 
 - model the probability, 
 - of y the true label, being equals to 1
 - given data/features x, 
 - in binary classification, since there are only two possible outcomes
 - probability of y = 0 would simply be 1 - p(y=1|x)

## Loss Function
### Let's say we have a probability ŷ and an actual label y, how do we measure how wrong our probability is?
Intuitively we can just measure the difference
Mean error: mean(ŷ - y)
Smaller difference = our model did a better job
 - but we would have negative and positive errors, 0 - 1 = -1 < 0.1-0 = 0.1 event though the first instance the prediction is further away

 - Two ways to solve negative problem: modulo the error
Mean absolute error: 
By taking absolute value of the error
So regardless of actual label being 0 or one, our error is only positive and comparable
 - squaring the error has a similar effect with some different nuances 
 - wont go into that here

### What's wrong with simply measuring the distance of our errors?
 - penalty grows linearly with error
 - which is acceptable, but there is another train of thought
 - which is to penalise a model dirproportionately harshly when it is confidently wrong
    - and this is not possible with a simple MAE

 - the other issue is more nuanced, 
 - where we first have to go through the whole model training process

### How do we train a log reg classification model?
### Training process of log reg,
### Forward pass, compute weighted sum, then sigmoid to get probability, then error
    - Compute weighted sum
    - Pass through sigmoid
    - get y_hat
    - compute loss against the true y, for now we just tryyy out MAE

### Backward propagation
Now that we have the loss, what's next?
To use this to improve the model, we want to nudge the learnable weights of the model, 
 - for each of these weights, we want to move them a tiny distance, in the direction that makes the overall loss go down (is overall right? is it loss per example or averaged loss of all examples)
 - To get the gradient, which is the direction to nudge each weight,
 - and this gradient, or direction, is usually unique to each weight, 
 - for example, y = passing rate with two weights for two features studying_time and tiktok_time, 
    - studying time makes more sense to have a positive weight, more studying = higher pass rate, so we want this weight to increase
    - tiktok_time's weight makes more sense to be negative, more tiktok, lower pass rate so we want the weight to decrease
    - you can see how you might want them to move in different directions to achieve the same goal

### Mathematical derivation of the loss function
BCE
CE
