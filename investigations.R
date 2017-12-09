library(caret)
library(dplyr)

set.seed(1)

# Loaded data
loaded <- read.csv("pml-training.csv", na.strings=c("", "NA", "na", "#DIV/0!"))

#
# Reduce dimenstions.
#

#' Title
#' @param df Data frame with all columsns.
#' @param na.threshold The max percentage of NAs to allow.
#' @return Data frame where all columns have NA percentage below the threshold.
selectColsWithData <- function(df, na.threshold=0.1) {
    colsWithData <- function(df) {
        nrows <- nrow(loaded)
        perc <- unlist(lapply(loaded, FUN = function(col) sum(is.na(col)) / nrows))
        names(perc[perc <= na.threshold])
    }
    select(df, colsWithData(df)) 
}

# Remove columns that are mostly NAs.
loaded <- 
    selectColsWithData(loaded, 0.1) %>% 
    select(-X, -user_name, -raw_timestamp_part_1, -raw_timestamp_part_2)

inTrain <- createDataPartition(y=loaded$classe, p=0.7, list=FALSE)
training <- loaded[inTrain,]
validation <- loaded[-inTrain,]

# Configure parallel processing to speed up RF training.
# See https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

# Use K-Fold cross-validation
fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

m1 <- train(classe ~ ., data=training, method="rf", trControl=fitControl)

stopCluster(cluster)
registerDoSEQ()

m1
m1$resample
confusionMatrix.train(m1)

# # Testing data....
# 
testing <- read.csv("pml-testing.csv")
# table(testing$classe)

table(predict(m1, testing))
predict(m1, testing)
# B A B A A E D B A A B C B A E E A B B B


