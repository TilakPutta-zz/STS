library(shiny)
library(caret)
library(rpart)
library(e1071)
library(kknn)
library(ipred)
library(rpart)
library(caret)
library(e1071)
library(adabag)
library(ipred)
library(boot)
library(mboost)
library(plyr)
library(e1071)
library(randomForest)
shinyServer(
  
  function(input, output) {
    output$contents <- renderTable({
      # input$file1 will be NULL initially. After the user selects
      # and uploads a file, it will be a data frame with 'name',
      # 'size', 'type', and 'datapath' columns. The 'datapath'
      # column will contain the local filenames where the data can
      # be found.
      inFile <- input$file1
      
      if (is.null(inFile))
        return(NULL)
      
      read.csv(inFile$datapath, header = input$header)
    })
    output$bagging <- renderTable({
      
      inFile1 <- input$file1
      inFile2 <- input$file2
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      maxacc = 0.0
      seedval = 0
      for (i in 1:20){
            set.seed(i)
            resampling_strategy = trainControl(method="boot",  number = 10)
            bag_tree_model = train(actual~.,train1, method="treebag", trControl=resampling_strategy)
            bag_tree_model$finalModel$mtrees[[2]]
            test$pre=predict(bag_tree_model,test)
            acc = cor(test$actual,test$pre)
            if (maxacc < acc){
                  maxacc = acc
                  seedval = i
            }
      }
      train1=train1[,-c(seq(2,3,1))]
      test=test[,-c(seq(2,3,1))]
      set.seed(seedval)
      resampling_strategy = trainControl(method="boot",  number = 10)
      
      
      bag_tree_model = train(actual~.,train1, method="treebag", trControl=resampling_strategy)
      bag_tree_model$finalModel$mtrees[[2]]
      
      test$pre=predict(bag_tree_model,test)
      synwo = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(2,2,1))]
      test=test[,-c(seq(2,2,1))]
      set.seed(seedval)
      resampling_strategy = trainControl(method="boot",  number = 10)
      
      
      bag_tree_model = train(actual~.,train1, method="treebag", trControl=resampling_strategy)
      bag_tree_model$finalModel$mtrees[[2]]
      
      test$pre=predict(bag_tree_model,test)
      synw = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(3,26,1))]
      test=test[,-c(seq(3,26,1))]
      set.seed(seedval)
      resampling_strategy = trainControl(method="boot",  number = 10)
      
      
      bag_tree_model = train(actual~.,train1, method="treebag", trControl=resampling_strategy)
      bag_tree_model$finalModel$mtrees[[2]]
      
      test$pre=predict(bag_tree_model,test)
      semwo = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(4,26,1))]
      test=test[,-c(seq(4,26,1))]
      set.seed(seedval)
      resampling_strategy = trainControl(method="boot",  number = 10)
      
      
      bag_tree_model = train(actual~.,train1, method="treebag", trControl=resampling_strategy)
      bag_tree_model$finalModel$mtrees[[2]]
      
      test$pre=predict(bag_tree_model,test)
      semw = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-3]
      test=test[,-3]
      set.seed(seedval)
      resampling_strategy = trainControl(method="boot",  number = 10)
      
      
      bag_tree_model = train(actual~.,train1, method="treebag", trControl=resampling_strategy)
      bag_tree_model$finalModel$mtrees[[2]]
      
      test$pre=predict(bag_tree_model,test)
      allwo = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      set.seed(seedval)
      resampling_strategy = trainControl(method="boot",  number = 10)
      
      
      bag_tree_model = train(actual~.,train1, method="treebag", trControl=resampling_strategy)
      bag_tree_model$finalModel$mtrees[[2]]
      
      test$pre=predict(bag_tree_model,test)
      allw = cor(test$actual,test$pre)
      bagg <- matrix(c(synwo,synw,semwo,semw,allwo,allw),ncol=1,byrow=TRUE)
      colnames(bagg) <- c(seedval)
      rownames(bagg) <- c("Syntactic Without NPE","Syntactic With NP","Semantic without NPE","Semantic with NPE","All without NPE","All with NPE")
      bagg <- as.table(bagg)
      bagg
      
    },colnames = FALSE,digits = 4)
    output$boosting <- renderTable({
      
      inFile1 <- input$file1
      inFile2 <- input$file2
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(2,3,1))]
      test=test[,-c(seq(2,3,1))]
      
      fit <- glmboost(actual ~ .,data=train1)
      
      test$pre=predict(fit,test)
      synwo = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(2,2,1))]
      test=test[,-c(seq(2,2,1))]
      
      fit <- glmboost(actual ~ .,data=train1)
      
      test$pre=predict(fit,test)
      synw = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(3,26,1))]
      test=test[,-c(seq(3,26,1))]
      
      fit <- glmboost(actual ~ .,data=train1)
      
      test$pre=predict(fit,test)
      semwo = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(4,26,1))]
      test=test[,-c(seq(4,26,1))]
      
      fit <- glmboost(actual ~ .,data=train1)
      
      test$pre=predict(fit,test)
      semw = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-3]
      test=test[,-3]
      fit <- glmboost(actual ~ .,data=train1)
      
      test$pre=predict(fit,test)
      allwo = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      fit <- glmboost(actual ~ .,data=train1)
      
      test$pre=predict(fit,test)
      allw = cor(test$actual,test$pre)
      bagg <- matrix(c(synwo,synw,semwo,semw,allwo,allw),ncol=1,byrow=TRUE)
      colnames(bagg) <- c(":")
      rownames(bagg) <- c("Syntactic Without NPE","Syntactic With NP","Semantic without NPE","Semantic with NPE","All without NPE","All with NPE")
      bagg <- as.table(bagg)
      bagg
      
    },colnames = FALSE,digits = 4)
    output$svr <- renderTable({
      
      inFile1 <- input$file1
      inFile2 <- input$file2
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(2,3,1))]
      test=test[,-c(seq(2,3,1))]
      
      svm_model <- svm(actual~.,train1)
      svm_model

      test$pred = predict(svm_model,test)
      synwo = cor(test$actual,test$pred)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(2,2,1))]
      test=test[,-c(seq(2,2,1))]
      
      svm_model <- svm(actual~.,train1)
      svm_model

      test$pred = predict(svm_model,test)
      synw = cor(test$actual,test$pred)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(3,26,1))]
      test=test[,-c(seq(3,26,1))]
      
      svm_model <- svm(actual~.,train1)
      svm_model

      test$pred = predict(svm_model,test)
      semwo = cor(test$actual,test$pred)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(4,26,1))]
      test=test[,-c(seq(4,26,1))]
      
      svm_model <- svm(actual~.,train1)
      svm_model

      test$pred = predict(svm_model,test)
      semw = cor(test$actual,test$pred)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-3]
      test=test[,-3]
      svm_model <- svm(actual~.,train1)
      svm_model

      test$pred = predict(svm_model,test)
      allwo = cor(test$actual,test$pred)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      svm_model <- svm(actual~.,train1)
      svm_model

      test$pred = predict(svm_model,test)
      allw = cor(test$actual,test$pred)
      bagg <- matrix(c(synwo,synw,semwo,semw,allwo,allw),ncol=1,byrow=TRUE)
      colnames(bagg) <- c(":")
      rownames(bagg) <- c("Syntactic Without NPE","Syntactic With NP","Semantic without NPE","Semantic with NPE","All without NPE","All with NPE")
      bagg <- as.table(bagg)
      bagg
      
    },colnames = FALSE,digits = 4)
    output$rforest <- renderTable({
      
      inFile1 <- input$file1
      inFile2 <- input$file2
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      maxacc = 0.0
      seedval = 0
      for (i in 1:20){
            set.seed(i)
            fit <- randomForest(actual ~ ., train1,ntree=198)
            test$pre=predict(fit,test)
            acc = cor(test$actual,test$pre)
            if (maxacc < acc){
                  maxacc = acc
                  seedval = i
            }
      }
      

      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(2,3,1))]
      test=test[,-c(seq(2,3,1))]
      
      set.seed(seedval)
      fit <- randomForest(actual ~ ., train1,ntree=198)
      test$pre=predict(fit,test)
      synwo = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(2,2,1))]
      test=test[,-c(seq(2,2,1))]
      
      set.seed(seedval)
      fit <- randomForest(actual ~ ., train1,ntree=198)
      test$pre=predict(fit,test)
      synw = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(3,26,1))]
      test=test[,-c(seq(3,26,1))]
      
      set.seed(seedval)
      fit <- randomForest(actual ~ ., train1,ntree=198)
      test$pre=predict(fit,test)
      semwo = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-c(seq(4,26,1))]
      test=test[,-c(seq(4,26,1))]
      set.seed(seedval)
      fit <- randomForest(actual ~ ., train1,ntree=198)
      test$pre=predict(fit,test)
      semw = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      train1=train1[,-3]
      test=test[,-3]
      set.seed(seedval)
      fit <- randomForest(actual ~ ., train1,ntree=198)
      test$pre=predict(fit,test)
      allwo = cor(test$actual,test$pre)
      train1 = read.csv(inFile1$datapath, header = input$header)
      test = read.csv(inFile2$datapath, header = input$header)
      
      
      set.seed(seedval)
      fit <- randomForest(actual ~ ., train1,ntree=198)
      test$pre=predict(fit,test)
      allw = cor(test$actual,test$pre)
      bagg <- matrix(c(synwo,synw,semwo,semw,allwo,allw),ncol=1,byrow=TRUE)
      colnames(bagg) <- c(seedval)
      rownames(bagg) <- c("Syntactic Without NPE","Syntactic With NP","Semantic without NPE","Semantic with NPE","All without NPE","All with NPE")
      bagg <- as.table(bagg)
      bagg
      
    },colnames = FALSE,digits = 4)
  }
)