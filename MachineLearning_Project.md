---
title: "Machine Learning Project"
author: "Venkata Muttineni"
date: "December 27, 2017"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

### 1. Title

predict the manner in which they did the exercise among belt, forearm, arm, and dumbell.

### 2. Synopsys

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

### 3. Data Loading and Processing

```{r message=FALSE}
### Load Packages
library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(gbm)

```
```{r}
### Download the Data
TrainData <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),header=TRUE)
TestData <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),header=TRUE)

dim(TrainData)
dim(TestData)

```

```{r results='hide'}
str(TrainData)
str(TestData)
```
The training data set is made of 19622 observations on 160 columns, Test data set is made of 20 observations on 160 columns. We can notice that many columns have NA values or blank values . So we will remove them, because they will not produce any information. 

```{r}

# Here we get the indexes of the columns having at least 90% of NA or blank values on the training dataset
indexColRemove <- which(colSums(is.na(TrainData) |TrainData=="")>0.9*dim(TrainData)[1]) 
TrainDataClean<- TrainData[,-indexColRemove]
TrainDataClean <- TrainDataClean[,-c(1:7)]
dim(TrainDataClean)

```

```{r}

# We do the same for the test set
indexColRemove <- which(colSums(is.na(TestData) |TestData=="")>0.9*dim(TestData)[1]) 
TestDataClean <- TestData[,-indexColRemove]
TestDataClean <- TestDataClean[,-1]
dim(TestDataClean)

```
After cleaning, now we have training data set has 53 columns and testing data set has 59 columns

```{r}
# Create a partition of training data set

set.seed(12345)
inTrain <- createDataPartition(TrainDataClean$classe, p=0.75, list=FALSE)
Train <- TrainDataClean[inTrain,]
Test <- TrainDataClean[-inTrain,]
dim(Train)
dim(Test)
```

***Train*** data set has ***14718*** observations on ***53*** columns and ***Test*** data set has ***4904*** observations on ***53*** columns 

### 4. Data Modeling

In the following sections, we will test 3 different models : ** classification tree **, **random forest **, **gradient boosting method**

In order to limit the effects of overfitting, and improve the efficicency of the models, we will use the **cross-validation** technique. We will use **5** folds (usually, **5** or **10** can be used, but **10** folds gives higher run times with no significant increase of the accuracy).

#### 4.1 Classification Tree - Train data set

```{r}
trnControl <- trainControl(method="cv", number=5)
ClassificationTreeModel <- train(classe~., data=Train, method="rpart", trControl=trnControl)

#print(ClassificationTreeModel)
fancyRpartPlot(ClassificationTreeModel$finalModel)

```

```{r}
trainPred <- predict(ClassificationTreeModel,newdata=Test)

confMatClassTree <- confusionMatrix(Test$classe,trainPred)

# display confusion matrix and model accuracy
confMatClassTree$table
```

```{r}
confMatClassTree$overall[1]
```

We can notice that the accuracy of this first model is very low (about **55%**). This means that the outcome class will not be predicted very well by the other predictors.

#### 4.2 Random Forests - Train data set

```{r}

RandomForestModel <- train(classe~., data=Train, method="rf", trControl=trnControl, verbose=FALSE)

print(RandomForestModel)
```


```{r}
plot(RandomForestModel,main="Accuracy of Random forest model by number of predictors")
```

```{r}
trainPred <- predict(RandomForestModel,newdata=Test)

confMatRandomForest <- confusionMatrix(Test$classe,trainPred)

# display confusion matrix and model accuracy
confMatRandomForest$table
```

```{r}
confMatRandomForest$overall[1]
```

```{r}
names(RandomForestModel$finalModel)
```

```{r}
RandomForestModel$finalModel$classes
```

```{r}
plot(RandomForestModel$finalModel,main="Model error of Random forest model by number of trees")

```

```{r}
# Compute the variable importance 
MostImpVars <- varImp(RandomForestModel)
MostImpVars

```

With random forest, we reach an accuracy of **99.2%** using cross-validation with 5 steps. This is very good. We can also Gradient boosting results.

We can also notice that the optimal number of predictors, i.e. the number of predictors giving the highest accuracy, is 27. There is no significal increase of the accuracy with 2 predictors and 27, but the slope decreases more with more than 27 predictors (even if the accuracy is still very good). The fact that not all the accuracy is worse with all the available predictors lets us suggest that there may be some dependencies between them.

At last, using more than about 30 trees does not reduce the error significantly.

#### 4.3 Gradient Boosting Method - Train data set

```{r}
GBModel <- train(classe~., data=Train, method="gbm", trControl=trnControl,verbose=FALSE)

print(GBModel)

```
```{r}
plot(GBModel)
```

```{r}
trainPred <- predict(GBModel,newdata=Test)

confMatGBModel <- confusionMatrix(Test$classe,trainPred)
confMatGBModel$table

```

```{r}
confMatGBModel$overall[1]

```
Precision with 5 folds is **96.3%**.

### 5. Conclusion

From above three methods, **Random Forest** model is the best one. We will then use it to predict the values of classe for the test data set.

```{r}
FinalTestPred <- predict(RandomForestModel,newdata=TestDataClean)
FinalTestPred
```