# Loading the library
library(glmnet)

# Loading the data
DATA_FILE_PATH = "../data/processed.csv"

df <- read.csv(DATA_FILE_PATH)
print(colnames(df))
x = model.matrix(value~., df)[,-1]
y = df$value
lambda = 10^seq(10, -2, length = 100)

set.seed(489)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
ytest = y[test]

#OLS
swisslm = lm(value~., data = df)
coef(swisslm)

swisslm <- lm(value~., data = df, subset = train)
ridge.mod <- glmnet(x[train,], y[train], alpha = 0)
#find the best lambda from our list via cross-validation
cv.out <- cv.glmnet(x[train,], y[train], alpha = 0)
cv.out

bestlam <- cv.out$lambda.min

#make predictions
ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test,])
s.pred <- predict(swisslm, newdata = df[test,])
#check MSE
print("The square error on the test set for a (normal) linear regression is:")
print(mean((s.pred-ytest)^2))

print("The square error on the test set for a ridge regression is:")
print(mean((ridge.pred-ytest)^2))
