setwd("/Users/ricardofernandez/Projects/DataScience/datasciencecoursera/Practical Machine Learning/Project")

# Define a seed so the process can be repeted
set.seed(12345)

library(caret)
library(corrplot)

if (!file.exists("pml-training.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
}

# Loadgin both training and testing datasets
trainingSet <- read.csv("pml-training.csv", header=TRUE, na.strings=c("", "NA", "NULL"))
testingSet<- read.csv("pml-testing.csv", header=TRUE, na.strings=c("", "NA", "NULL"))

# Checking set sizes
dim(trainingSet)
dim(testingSet)

# We need to check the training dataset looking for values that we need to clean
str(trainingSet)

# We can observe that there are some columns with NA's values, we will define a
# threshold and clean all the columns that have too much NA values
threshold = 0.95
trainingNoNA <- trainingSet[, (nrow(trainingSet)-colSums(is.na(trainingSet)))/nrow(trainingSet) > threshold]
# Checking set size
dim(trainingNoNA)

# Remove unrelevant variables unlikely to be related to dependent variable.
trainingClean <- trainingNoNA[,-c(1:7)]
dim(trainingClean)

# Data spliting for trainign and Checking correlation
triningPartition <- createDataPartition(trainingClean$classe, p = 0.7, list=FALSE)
trainSubSet <- trainingClean[triningPartition,]
crossValSubSet <- trainingClean[-triningPartition,]

# Make a correlation matrix plot
corMat <- cor(trainSubSet[,-dim(trainSubSet)[2]])
corrplot(corMat, method = "square", type="lower", order="AOE", tl.cex = 0.7, 
         tl.col="black", tl.srt = 45, diag = FALSE)

# Remove higly correlated variables
highlyCor <- findCorrelation(corMat, cutoff = 0.8)
trainNotCorr <- trainSubSet[,-highlyCor]

# Higly correlated variables
names(crossValSubSet)[highlyCor]
# Total variables 
ncol(trainNotCorr)

# New correlation matrix plot with non correlated variables
cormat <- cor(trainNotCorr[,-dim(trainNotCorr)[2]])
corrplot(cormat, method = "square", type="lower", order="AOE", tl.cex = 0.7, 
         tl.col="black", tl.srt = 45, diag = FALSE)

# Model using random forest
model <- train(classe~., 
                    method = "rf", 
                    data=trainNotCorr, 
                    trControl = trainControl(method = "cv"), importance=TRUE)

# Random Forest info
print(model, digits = 3)
# model$finalModel

# Plot the Predictors importance
varImpPlot(model$finalModel, main = "Predictors importance", pch = 20, cex = 0.8)

# Predict outcomes using validation set
prediction <- predict(model, crossValSubSet)

# Show prediction result
confMatrix <- confusionMatrix(crossValSubSet$classe, prediction)

# Show accuracy
accuracy <- confMatrix$overall[1]

# Finaly apply the model to the testing Set
predict(model, testingSet)
