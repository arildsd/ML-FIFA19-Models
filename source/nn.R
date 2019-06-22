# File path to the data
DATA_FILE_PATH = "../data/processed.csv"

# Load the pre-processed data
df = read.csv(DATA_FILE_PATH)

# Normalize the data
normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(df, normalize))

x = model.matrix(value~., maxmindf)[,-1]
y = df$value


set.seed(489)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
ytest = y[test]

library(neuralnet)

nn <- neuralnet(y ~ x,data=train, hidden=c(2,1), linear.output=TRUE, threshold=0.01)
plot(nn)
nn$result.matrix



