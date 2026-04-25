## LogReg output: σ(z) = sigmoid(ax+b)
### CrossEntropy loss for one row/example
(Binary)
= -[ylog(sigma(z)) + (1-y)log(1-sigma(z))]
 - where y = 1
 - = -log(output)
 - = -log(σ(z))
 - = -log(1/(1+e^-z))

### When it breaks:
1) when computing logreg output
2) eg. z = -100, e^100 is super big,
3) for float 32 it overflows and calls it inf
    - [elborate: bounds of float 32 etc]
4) infinity in denominator, sigmoid(z) = 0.0
5) loss = -log(0.0) = -inf
6) loss inf value: taining breaks

### 2 Solutions:
### 1) Clipping
Relace 0.0 before taking the log in steps 4/5
 - eg. sklearn: ses clipping directly. ŷ values are clipped to [eps, 1-eps] where eps is the machine precision for ŷ's dtype
 - [elaborate: "machine precision for y_hatos dtype"]
 - works but this is slightly falsifying value

### 2) log-sum-exp trick
 - -log(sigma(z))
 - -log(1/(1+e^-z))
 - log(1+e^-z)

no clipping or approximation
 - arrive at the same mathematical (more precise than clipping)
 - via a route where no intermediate step
 - produced an astronomically large/small number
