# Logistic Regression


Have implemented the logistic regression for all the 3 types.

## Binary

Binary logistic regression is used to model the relationship between a binary dependent variable (with only two possible outcomes) and one or more independent variables. It estimates the probability of the dependent variable taking on a particular value (typically 1) based on the values of the independent variables. The model uses the logistic function to transform a linear combination of the independent variables into a probability, which is then used to predict the outcome of the dependent variable. 

## Multinomial

Multinomial logistic regression can be thought of as training multiple binary logistic regression models simultaneously, with each model comparing one category against all others. For example, if we have a dependent variable with three categories (A, B, and C), we would train three separate binary logistic regression models: A vs. (B and C), B vs. (A and C), and C vs. (A and B).

During inference, we apply each of the three trained models to the input and obtain three probability scores for each of the three categories. These probabilities are combined to produce the final prediction for the input. The category with the highest probability is selected as the predicted outcome.

The advantage of multinomial logistic regression over training multiple binary logistic regression models separately is that the models share information about the relationship between the dependent variable and the independent variables, which can lead to improved predictions.

## Ordinal

Ordinal logistic regression is a statistical technique used for predicting an ordinal outcome variable, which has three or more levels that are ordered but not equally spaced. The technique is an extension of binary logistic regression, which is used for predicting binary outcomes.

In ordinal logistic regression, the dependent variable is modeled as a function of one or more independent variables using a cumulative probability function. The model estimates the probability that the dependent variable falls into each of the ordinal categories given the values of the independent variables. The coefficients in the model represent the log odds of being in a higher category compared to a lower category, given the values of the independent variables.

### Model Representation

The logistic regression model can be represented mathematically as:

$$ P(y=1|x_1,x_2,...,x_n) = \frac{1}{1+e^{-z}} $$

where, \
$P(y=1|x_1,x_2,...,x_n)$ is the probability that the dependent variable (y) is 1 given the values of the independent variables (x1, x2,...,xn) \
$z$ is the linear combination of the independent variables and their coefficients, given by:
$$ z = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n $$
$ b_1 - b_n $ are the weights given to the input features $ x_1 - x_n $ respectively. $b_0$ is the intercept.

We see that everything is similar except giving the output to sigmoid function to compute the probability and classify.


### Cost Function

The cost function for logistic regression is defined as:

$$ L(\boldsymbol{b}) = -\frac{1}{m} \Sigma_{i=1}^m [y_i log(\hat{y_i}) + (1-y_i)log(1-\hat{y_i})] $$

This is also called as the logistic loss function or the cross-entropy loss function.

where:

$m$ is the number of training examples \
$\boldsymbol{b}$ is the vector of coefficients for the independent variables \
$y_{i}$ is the true label for the $i$-th training example \
$\hat{y}_{i}$ is the predicted probability for the $i$-th training example 

The first term in the cost function measures the error when the true label is 1, and the second term measures the error when the true label is 0. When the predicted probability $\hat{y}_{i}$ is close to the true label $y_{i}$, the cost function will be small. However, when the predicted probability is far from the true label, the cost function will be large.

The goal of logistic regression is to find the set of coefficients $\boldsymbol{b}$ that minimize the cost function $J(\boldsymbol{b})$. We do this using optimization techniques such as gradient descent which was explained in Linear Regression.


### Sigmoid function

Sigmoid function is used to model the relationship between the independent variables and the dependent variable. 

As discussed in the model representation, sigmoid function is as follows:

$$ \sigma(z) = \frac{1}{1+e^{-z}} $$

where $z$ is a linear combination of the independent variables and their coefficients:
$$ z = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n $$

The sigmoid function takes the input value $z$ and maps it to a value between 0 and 1, which represents the probability that the dependent variable is 1 given the values of the independent variables. When $z$ is positive, the sigmoid function outputs a value closer to 1, and when $z$ is negative, the sigmoid function outputs a value closer to 0.

The sigmoid function has an S-shaped curve, which allows it to model nonlinear relationships between the independent variables and the dependent variable. In logistic regression, the goal is to find the set of coefficients $\beta_0, \beta_1, \beta_2, \cdots, \beta_n$ that maximize the likelihood of the observed data. The sigmoid function is used to compute the predicted probabilities, which are then compared to the true labels to compute the likelihood of the observed data.

Following is a visualization graph for sigmoid function.
[Sigmoid.png])(https://github.com/ANIRUDH-333/Logistic-Regression/blob/main/sigmoid.png)

