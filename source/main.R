# Title     : TODO
# Objective : TODO
# Created by: Oreo
# Created on: 31 May 2019

print("Starting model training")

# Loading the library
library(glmnet)
library(rpart)
library(rpart.plot)

# File path to the data
DATA_FILE_PATH = "../data/processed.csv"

# Load the pre-processed data
df = read.csv(DATA_FILE_PATH)

#print(colnames(df))
x = model.matrix(value~., df)[,-1]
y = df$value
# Makes a sequence of lambda from 10^10 to 10^-2
lambda = 10^seq(10, -2, length = 100)

set.seed(489)
train = sample(1:nrow(x), nrow(x)*0.7)
test = (-train)
ytest = y[test]


# Normal linear models
#OLS
linear.model = lm(value~., data = df)
coef(linear.model)

linear.model = lm(value~., data = df, subset = train)
s.pred = predict(linear.model, newdata = df[test,])

# RIDGE
#find the best lambda from our list via cross-validation
cv.out = cv.glmnet(x[train,], y[train], alpha = 0, nfolds = 10, keep = TRUE)
ridge.mod = glmnet(x[train,], y[train], alpha = 0, standardize=FALSE)

ridge_bestlam = cv.out$lambda.min
print("Best lambda was:")
print(ridge_bestlam)

ridge.pred = predict(ridge.mod, s = ridge_bestlam, newx = x[test,])

# LASSO
#find the best lambda from our list via cross-validation
cv.out = cv.glmnet(x[train,], y[train], alpha = 1, nfolds = 10)
lasso.mod = glmnet(x[train,], y[train], alpha = 1, standardize=FALSE)

lasso_bestlam = cv.out$lambda.min
print("Best lambda was:")
print(lasso_bestlam)

#make predictions
lasso.pred = predict(lasso.mod, s = lasso_bestlam, newx = x[test,])

#check MSE
print("The square error on the test set for a (normal) linear regression is:")
print(mean((s.pred-ytest)^2))

print("The square error on the test set for a ridge regression is:")
print(mean((ridge.pred-ytest)^2))

print("The square error on the test set for a lasso regression is:")
print(mean((lasso.pred-ytest)^2))

min.error = 1.0e+300 # Initalize to the maximum possible value for the machine
best.alpha = -1
best.lambda = -1
# Try with different values for alpha and lambda for a general linear model
for (alpha in seq(0, 1, length = 11)){
    result = cv.glmnet(x[train,], y[train], alpha = alpha, nfolds = 10)

    # Get min cv error
    error = min(result$cvm)
    if (error < min.error){
        min.error = error
        best.alpha = alpha
        best.lambda = result$lambda.min
    }
}

print("The best alpha is:")
print(best.alpha)
print("The error for this alpha in cross validation was:")
print(min.error)


# Make a general linear model with the optimal lambda and alpha
general.mod = glmnet(x[train,], y[train], alpha = best.alpha, standardize=FALSE)
#make predictions
general.pred = predict(general.mod, s = best.lambda, newx = x[test,])


print("The square error on the test set for a glm regression with is:")
print(mean((general.pred-ytest)^2))

# NON LINEAR MODELS

print("///////////////////DECISION TREES///////////////////")
# Decision trees
# Pick hyper parameters with 10 fold cross validation
best_depth = -1
min_cv_error = 1.0e+300
for (depth in 2:20){
    control = rpart.control(cp=0.001, maxdepth = 10) # Set hyper-parameters for the tree fiting
    fit = rpart(value~., data = df, subset = train, method = "anova", control = control)

}

printcp(fit) # display the results

plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

# plot tree
rpart.plot(fit, type = 3, digits = 3, fallen.leaves = TRUE)

# Predict
tree_pred = predict(fit, test)
print(mean((tree_pred-ytest)^2))



print("Model training ended")