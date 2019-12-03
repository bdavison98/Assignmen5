library(caret)
library('RANN')
library('dplyr')
library("randomForest")
library(gbm)
library("klaR")
library(ggplot2)
library(gridExtra)


data("scat")
sum(is.na(scat))


scat_categorical <- scat


# QUESTION 1 convert species to numeric

  scat$Species <- ifelse(scat$Species=="coyote",1, ifelse(scat$Species == "bobcat", 2,ifelse(scat$Species=="gray_fox",3,0)))


# QUESTION 2 in Remove the Month, Year, Site, Location features
    
  dplyr::select (scat, -c("Month", "Year", "Site", "Location"))
  
  dplyr::select (scat_categorical, -c("Month", "Year", "Site", "Location"))
  
# QUESTION 3 imputed missing values
  
  preProcValues <- preProcess(scat, method = c("knnImpute","center","scale"))
  
  scat_processed_Categorical <- predict(preProcValues, scat_categorical)
  
  scat_processed <- predict(preProcValues, scat)
  
  sum(is.na(scat_processed))

  
# QUESTION 4
  
  
# There is no categorical data to convert


# QUESTION 5
  
  set.seed(100)
  index <- createDataPartition(scat_processed$Species, p=0.75, list=FALSE)
  
  trainSet <- scat_processed[ index,]
  
  testSet <- scat_processed[-index,]
  
  control <- rfeControl(functions = rfFuncs,
                        method = "repeatedcv",
                        repeats = 3,
                        verbose = FALSE)
  
  outcomeName<-'Species'
  
  predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
  
  Species_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],rfeControl = control)
  
  Species_Pred_Profile
  
  predictors<-c("d15N", "d13C", "Mass", "CN", "Site")
  
  
  fitControl <- trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 5)
  
  ##################### GBM #####################
  
  model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm', importance=T)
  
  grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
  
  model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneGrid=grid, importance=T)
  
  print(model_gbm)
  
  plot(model_gbm)
  
  predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
  
  table(predictions)
  
  confusionMatrix(table(factor(predictions_gbm, levels=1:27),factor(testSet$Species, levels=1:27)))
  
  ##################### RAND FOREST #####################
  
  model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf', importance=T)
    
  grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
    
  model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',trControl=fitControl,tuneGrid=grid, importance=T)
    
  print(model_rf)
    
  plot(model_rf)
  
  predictions_rf<-predict.train(object=model_rf,testSet[,predictors],type="raw") 
  
  confusionMatrix(table(factor(predictions_rf, levels=1:27),factor(testSet$Species, levels=1:27)))
  
  ##################### Neural Net. #####################
  
  model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet', importance=T)
  
  grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
  
  model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet',trControl=fitControl,tuneGrid=grid)
  
  print(model_nnet)
  
  plot(model_nnet)
  
  predictions_nnet<-predict.train(object=model_nnet,testSet[,predictors],type="raw") 
  
  confusionMatrix(table(factor(predictions_nnet, levels=1:27),factor(testSet$Species, levels=1:27)))
  predictions_nnet
  
  ##################### NAIVE BAYES #####################
  
  index_categorical <- createDataPartition(scat_processed_Categorical$Species, p=0.75, list=FALSE)
  
  set.seed(100)
  
  trainSet_categorical <- scat_processed_Categorical[ index_categorical,]
  
  testSet_categorical <- scat_processed_Categorical[-index_categorical,]
  
  model_bayes<-train(trainSet_categorical[,predictors],trainSet_categorical[,outcomeName],method='naive_bayes', importance=T)
  
  grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
  
  model_bayes<-train(trainSet_categorical[,predictors],trainSet_categorical[,outcomeName],method='naive_bayes',trControl=fitControl,tuneGrid=grid)
  
  print(model_bayes)
  
  predictions_bayes<-predict.train(object=model_bayes,testSet_categorical[,predictors],type="raw") 
  
  confusionMatrix(table(factor(predictions_bayes, levels=1:27),factor(testSet_categorical$Species, levels=1:27)))
  
  ##################### Var. Importance #####################

  varImp(object=model_gbm)

  gbmPLot <- plot(varImp(object=model_gbm),main="GBM - Variable Importance")
  
  varImp(object=model_rf)

  rfPlot <- plot(varImp(object=model_rf),main="RF - Variable Importance")
  
  varImp(object=model_nnet)

  netPlot <- plot(varImp(object=model_nnet),main="NNET - Variable Importance")
  
  varImp(object=model_bayes)

  bayesPlot <- plot(varImp(object=model_bayes),main="Bayes - Variable Importance")

# QUESTION 6
 

# QUESTION 7

  model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)

  print(model_gbm)
  
  plot(model_gbm)

# Question 8

  grid.arrange(gbmPLot, rfPlot,netPlot, bayesPlot,nrow = 2)
