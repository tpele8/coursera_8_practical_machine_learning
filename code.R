###### Clearing workspace ######
rm(list=ls())
cat("\014")

# Import libraries
library(caret)
library(rpart)
library(randomForest)
library(plyr)
library(gbm)
library(survival)
library(splines)
library(parallel)

# Constants:
READFILETRAIN <- '/Users/i64425/Documents/Coursera/coursera_8_practical_machine_learning/Course Project/Read/pml-training.csv'
READFILETEST <- '/Users/i64425/Documents/Coursera/coursera_8_practical_machine_learning/Course Project/Read/pml-testing.csv'
WRITEPLOT <- '/Users/i64425/Documents/Coursera/coursera_8_practical_machine_learning/Course Project/Write/EDA_plots.png'
WRITEFILE1 <- '/Users/i64425/Documents/Coursera/coursera_8_practical_machine_learning/Course Project/Write/df_for_research.csv'
WRITEFILE2 <- '/Users/i64425/Documents/Coursera/coursera_8_practical_machine_learning/Course Project/Write/df_for_submission.csv'

# Read Data in:
dfTrainLoad<- read.csv(READFILETRAIN, header = TRUE)
dfTestLoad <- read.csv(READFILETEST, header = TRUE)

# Manipulate df to only contain relevant data:
vars <- c('X', 'classe')
varList <- as.list(grep("accel", names(dfTrainLoad), value = TRUE))
finalVars <- c(varList, vars)
dfFinal <- dfTrainLoad[,names(dfTrainLoad) %in% finalVars]

# Investigate columns with ambiguous meaning:
write.csv(dfFinal, WRITEFILE1)
str(dfFinal$var_total_accel_belt)
str(dfFinal$var_accel_arm)
str(dfFinal$var_accel_forearm)
str(dfFinal$var_accel_dumbbell)

beltVarDF <- as.data.frame(table(dfFinal$var_total_accel_belt))
beltVarNA <- sum(is.na(dfFinal$var_total_accel_belt))
beltVarNoNA <- sum(!is.na(dfFinal$var_total_accel_belt))
beltVarNAPerc <- beltVarNA/(beltVarNoNA + beltVarNA)
beltVarNoNAPerc <- beltVarNoNA/(nrow(dfFinal))

armVarDF <- as.data.frame(table(dfFinal$var_accel_arm))
armVarNA <- sum(is.na(dfFinal$var_accel_arm))
armVarNoNA <- sum(!is.na(dfFinal$var_accel_arm))
armVarNAPerc <- armVarNA/(nrow(dfFinal))
armVarNoNAPerc <- armVarNoNA/(nrow(dfFinal))

forearmVarDF <- as.data.frame(table(dfFinal$var_accel_forearm))
forearmVarNA <- sum(is.na(dfFinal$var_accel_forearm))
forearmVarNoNA <- sum(!is.na(dfFinal$var_accel_forearm))
forearmVarNAPerc <- forearmVarNA/(nrow(dfFinal))
forearmVarNoNAPerc <- forearmVarNoNA/(nrow(dfFinal))

dumbbellVarDF <- as.data.frame(table(dfFinal$var_accel_dumbbell))
dumbbellVarNA <- sum(is.na(dfFinal$var_accel_dumbbell))
dumbbellVarNoNA <- sum(!is.na(dfFinal$var_accel_dumbbell))
dumbbellVarNAPerc <- dumbbellVarNA/(nrow(dfFinal))
dumbbellVarNoNAPerc <- dumbbellVarNoNA/(nrow(dfFinal))

measureTable = data.frame('Measure_Name' = c('beltVarNAPerc', 'armVarNAPerc', 'forearmVarNAPerc', 'dumbbellVarNoNAPerc'), 
                          'Measure_Value' = c(beltVarNAPerc, armVarNAPerc ,forearmVarNAPerc, dumbbellVarNoNAPerc))
print(measureTable)

dfFinalTemp <- dfFinal[complete.cases(dfFinal),]

png(filename = WRITEPLOT)
par(mfrow= c(2,2))
plot(dfFinal$var_accel_arm)
plot(dfFinal$var_accel_dumbbell)
plot(dfFinal$var_accel_forearm)
plot(dfFinal$var_total_accel_belt)
dev.off()

# NOTE: After significant investigation into the dataset, it has been determined that the
#       var_* columns can be removed from the dfFinal dataset
varVars <- as.list(grep("var", names(dfFinal), value = TRUE))
dfFinal2 <- dfFinal[,!names(dfFinal) %in% varVars]
dfFinal2$classe <- factor(dfFinal2$classe)

# Split data into train and test datasets:
set.seed(1234)
inTraining <- createDataPartition(dfFinal$classe, p = 0.7, list = FALSE)
dfTrain <- dfFinal2[inTraining, ]
dfTest <- dfFinal2[-inTraining, ]

# Run different models on the data:
# Data Trees:
treeMod <- rpart(classe ~ ., data = dfTrain, method = 'class')
treePred <- predict(treeMod, dfTest, type = 'class')
treeCM <- confusionMatrix(treePred, dfTest$classe)

print(paste('Data tree method overall accuracy: ', treeCM$overall[1]))
print(treeCM$table)

# Random Forest:
rfMod <- randomForest(classe ~ ., data = dfTrain, na.action = na.exclude)
rfPred <- predict(rfMod, dfTest, type = 'class', na.action = na.exclude)
rfCM <- confusionMatrix(rfPred, dfTest$classe)

print(paste('Random Forest method overall accuracy: ', rfCM$overall[1]))
print(rfCM$table)

# Gradient Boosting Method:
gbmMod <- gbm(classe ~ ., distribution = 'multinomial', data = dfTrain, cv.folds = 5, verbose = FALSE)
gbmPred <- as.data.frame(predict(gbmMod, newdata = dfTest))
names(gbmPred) <- c('A', 'B', 'C', 'D', 'E')
gbmPredVals <- colnames(gbmPred)[apply(gbmPred, 1, which.max)]

gbmCM <- confusionMatrix(gbmPredVals, dfTest$classe)
print(paste('GBM overall accuracy: ', gbmCM$overall[1]))
print(gbmCM$table)

# Comparing the confusion matrices and the accuracy tables, it's a tossup between the GBM and Data Trees methods.
# The computation time wasn't significantly different between the two and, given GBM's general ability 
# to exceed Data Trees in prediction, we will be using that method to predict the output for the test case.

testPredict <- as.data.frame(predict(gbmMod, dfTestLoad))
names(testPredict) <- c('A', 'B', 'C', 'D', 'E')

classe <- colnames(testPredict)[apply(testPredict, 1, which.max)]

# Bind predictions to testing dataframe
testPredictFinal <- cbind(dfTestLoad, classe)

# Add description column so classes are described within the dataframe
testPredictFinal$classe_desc <- NA
for(i in 1:nrow(testPredictFinal)){
  if(testPredictFinal$classe[i] == 'A'){
    testPredictFinal$classe_desc[i] = 'Exactly according to the specification'
  } else if (testPredictFinal$classe[i] == 'B'){
    testPredictFinal$classe_desc[i] = 'Throwing the elbows to the front'
  } else if (testPredictFinal$classe[i] == 'C'){
    testPredictFinal$classe_desc[i] = 'Lifting the dumbbell only halfway'
  } else if (testPredictFinal$classe[i] == 'D'){
    testPredictFinal$classe_desc[i] = 'Lowering the dumbbell only halfway'
  } else if (testPredictFinal$classe[i] == 'E'){
    testPredictFinal$classe_desc[i] = 'Throwing the hips to the front'
  }
}

# Write resulting dataframe to file for submission
write.csv(testPredictFinal, WRITEFILE2)
testPredictFinal2 <- testPredictFinal[, names(testPredictFinal)%in%c('X', 'user_name', 'classe', 'classe_desc')]
