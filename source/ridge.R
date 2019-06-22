# Loading the library
library(glmnet)

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
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
ytest = y[test]

min.error = 1.0e+300 # Initalize to the maximum possilbe value for the machine
best.alpha = -1
for (alpha in seq(0, 1, length = 11)){
    result = cv.glmnet(x[train,], y[train], alpha = alpha, nfolds = 20, parallel = TRUE)


    glm_bestlam = result$lambda.min


    # compute square error
    error = result$cvm
    if (error < min.error){
        min.error = error
        best.alpha = alpha
    }
}

print("The best alpha is:")
print(best.alpha)
print("The error for this alpha was:")
print(min.error)
