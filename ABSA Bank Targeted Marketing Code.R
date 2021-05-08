#Exploratory Analysis of the Data set

#Importing the dataset
market <- read.csv("Market.csv")
head(market)

summary(market)

str(market)

#Converting the categorical variables into factors
market$gender <- as.factor(market$gender)
market$marital_status <- as.factor(market$marital_status)
market$education <- as.factor(market$education)
market$employ_status <- as.factor(market$employ_status)
market$spouse_work <- as.factor(market$spouse_work)
market$residential_status <- as.factor(market$residential_status)
market$product <- as.factor(market$product)
market$purchase <- as.factor(market$purchase)

#Information gain of the parameters
library(FSelector)
market_information_gain <- information.gain(purchase~., market)
market_information_gain %>% arrange(desc(attr_importance))

#DESCRIPTIVE ANALYTICS OF THE DATA
library(dplyr)

##############################  PRODUCT ########################################
#Relation of the product with its purchase
plot(market %>% filter(purchase=="yes") %>% select(product), 
     xlab = "Products", 
     ylab = "Purchases", 
     main = "Purchases Based on Product Category", 
     ylim = c(0,7000))

############################  AGE  ################################
#Filtering the data for customers who had purchased the products in the past
age_graph <- (market %>% filter(purchase=="yes") %>% select(age))
summary(age_graph)

#Finding absolute frequency
sort(table(age_graph))

#Plotting Age groups vs Purchases
hist(age_graph$age, xlab = "Age Groups", 
     ylab = "Purchases made", 
     main = "Number of Purchases Based on Age Groups", 
     ylim = c(0, 5000))

#Plotting products vs age groups
plot(market %>% filter(purchase=="yes") %>% select(product, age), ylab = "Age Groups", 
     xlab = "Products", 
     main = "Products Purchased Based on Age Groups")

############################  NUMBER OF CHILDREN  #################################
#Filtering the data for customers who had purchased the products in the past
children_graph <- (market %>% filter(purchase=="yes") %>% select(nb_depend_child))

summary(children_graph)

#Finding absolute frequency 
sort(table(children_graph))

#Plotting the relation between number of children and purchases
hist(children_graph$nb_depend_child, xlab = "Number of children", 
     ylab = "Purchases made", 
     main = "Purchases Based on Number of Children", 
     ylim = c(0, 8000))

######################  YEARS EMPLOYED  ##########################################
#Filtering the data for customers who had purchased the products in the past
employment_graph <- (market %>% filter(purchase=="yes") %>% select(yrs_employed))

summary(employment_graph)

#Finding absolute frequency
sort(table(employment_graph))

#Plotting the relation between number of years employed and purchases
hist(employment_graph$yrs_employed, xlab = "Employment Years", 
     ylab = "Purchases made", 
     main = "Purchases Based on Years of Employment", 
     ylim = c(0, 8000))

#################################  NET INCOME   ############################## 
#Filtering the data for customers who had purchased the products in the past
net_income_graph <- (market %>% filter(purchase=="yes") %>% select(net_income))

summary(net_income_graph)

#Finding absolute frequency
sort(table(net_income_graph))

#Plotting the relation between Net Income and purchases
hist(net_income_graph$net_income, xlab = "Net Income", 
     ylab = "Purchases made", 
     main = "Purchases Based on Net Income", 
     xlim = c(0,200000),
     ylim = c(0, 4000))

#Plotting the relation between number of years employed and purchases
plot(market %>% filter(purchase=="yes") %>% select(product, net_income))

#############################   MARITAL STATUS  ##########################
#Plotting the relation between marital status and purchases
plot(market %>% filter(purchase=="yes") %>% select(marital_status),
     xlab = "Marital Status",
     ylab = "Purchases made", 
     main = "Purchases Based on Marital Status", 
     ylim = c(0, 5000))

################  SPLITTING THE DATA INTO TRAINING AND TEST SETS  ##################
#Loading the required libraries
library(caret) #Classification and Regression Training
library(rpart) #Recursive partitioning and regression trees
library(rpart.plot) #Plotting the rpart model
library(e1071) #Naive Bayes
library(randomForest) #Random Forest Classification

#Splitting the data set into Training and Test Set
indxTrain <- createDataPartition(market$purchase, p=0.75, list=F)

train_data <- market[indxTrain,]
test_data <- market[-indxTrain,]

#############################  DECISION TREES  ################################

#Configuring the control settings
ctrl_dt <- rpart.control(minsplit = 1, minbucket = 1, maxdepth = 10, cp = 0.01)

#Building the learner model
market_dt <- rpart(purchase~., data = train_data, control = ctrl_dt)
market_dt

#Plotting the decision tree
rpart.plot(market_dt, main = "Market", extra = 108)

#Making predictions on the test data
predict_test_dt <- predict(market_dt, newdata = test_data, type = "class")
predict_test_dt

#Building the confusion matrix for Decision Tree Classification
confusionMatrix(predict_test_dt, factor(test_data$purchase))

#####################################  RANDOM FOREST   ########################

#Building the learner model
market_rf <- randomForest(purchase~., data=train_data, ntree = 500)
market_rf

#Making predictions on the test data
predict_test_rf <- predict(market_rf, newdata = test_data, type = "class")
predict_test_rf

#Building the confusion matrix
confusionMatrix(predict_test_rf, factor(test_data$purchase))

###################################  NAIVE BAYES  ############################
#Building the learner model
market_nb <- naiveBayes(purchase~., data=train_data)
market_nb

#Making predictions on the test data
predict_test_nb <- predict(market_nb, newdata = test_data, type = "class")
predict_test_nb

##Building the confusion matrix
confusionMatrix(predict_test_nb, factor(test_data$purchase))

#################################  SVM  ######################################
#Building the learner model
market_svm <- svm(purchase~., data=train_data)
market_svm

#Making predictions on the test data
predict_test_svm <- predict(market_svm, newdata = test_data, type = "class")
predict_test_svm

##Building the confusion matrix
confusionMatrix(predict_test_svm, factor(test_data$purchase))

########################  K-FOLD CROSS VALIDATION  ######################

#Applying k-Fold Cross Validation to Random Forest Model
#Creating the number of folds
folds_rf <- createFolds(train_data$purchase, k = 10)

#Creating the function to train and test the model on each fold of data
cv_rf <- lapply(folds_rf, function(x) {
        #Creating the training fold without test fold
        training_fold <- train_data[-x,]
        test_fold <- train_data[x,]
        #Building the learner model for each of the 10 unique training fold
        market_rf_kfold <- randomForest(purchase~., data=training_fold, ntree = 500)
        #Making predictions on each unique test fold
        predict_test_rf_kfold <- predict(market_rf_kfold, newdata = test_fold, type = "class")
        #Creating confusion matrix for each unique test fold
        cm_rf <- table(predict_test_rf_kfold, factor(test_fold$purchase))
        #Calculating accuracy for each fold
        accuracy_rf <- (cm_rf[1,1] + cm_rf[2,2]) / (cm_rf[1,1] + cm_rf[2,2] + cm_rf[1,2] + cm_rf[2,1])
        return(accuracy_rf)
}) 
cv_rf

#Calculating the mean of all the 10 accuracy values to determine the ultimate model accuracy
mean(as.numeric(cv_rf))
#Calculating the standard deviation of the 10 accuracy values to determine model stability
sd(as.numeric(cv_rf))

#--------------------------------------------------------------------------------------------

#Applying k-Fold Cross Validation to SVM Model
#Creating the number of folds
folds_svm <- createFolds(train_data$purchase, k = 10)

#Creating the function to train and test the model on each fold of data
cv_svm <- lapply(folds_svm, function(x) {
        #Creating the training fold without test fold
        training_fold <- train_data[-x,]
        test_fold <- train_data[x,]
        #Building the learner model for each of the 10 unique training fold
        market_svm_kfold <- svm(purchase~., data=training_fold)
        #Making predictions on each unique test fold
        predict_test_svm_kfold <- predict(market_svm_kfold, newdata = test_fold, type = "class")
        #Creating confusion matrix for each unique test fold
        cm_svm <- table(predict_test_svm_kfold, factor(test_fold$purchase))
        #Calculating accuracy for each fold
        accuracy_svm <- (cm_svm[1,1] + cm_svm[2,2]) / (cm_svm[1,1] + cm_svm[2,2] + cm_svm[1,2] + cm_svm[2,1])
        return(accuracy_svm)
}) 
cv_svm

#Calculating the mean of all the 10 accuracy values to determine the ultimate model accuracy
mean(as.numeric(cv_svm))
#Calculating the standard deviation of the 10 accuracy values to determine model stability
sd(as.numeric(cv_svm))

############################  FINAL PREDICTIONS  ######################################

#Reading the dataset for predictions
market_pred <- read.csv("market_pred.csv")

#Converting the categorical variables into factors
market_pred$gender <- as.factor(market_pred$gender)
market_pred$marital_status <- as.factor(market_pred$marital_status)
market_pred$education <- as.factor(market_pred$education)
market_pred$employ_status <- as.factor(market_pred$employ_status)
market_pred$spouse_work <- as.factor(market_pred$spouse_work)
market_pred$residential_status <- as.factor(market_pred$residential_status)
market_pred$product <- as.factor(market_pred$product)
market_pred$purchase <- as.factor(market_pred$purchase)

str(market_pred)

#Making the predictions using the model built using Random Forest Classification
market_pred_rf <- predict(market_rf, newdata = market_pred, type = "class")
market_pred_rf

#Replacing the target variable column in the market_pred dataset with the predictions
market_pred$purchase <- market_pred_rf
market_pred$purchase

###########################################################################################



####################  ANALYSING TRENDS IN PREDICTED DATA  ###############################
#Applying k-Fold Cross Validation to Decision Tree Model
#Creating the number of folds
folds_dt <- createFolds(train_data$purchase, k = 10)

#Creating the function to train and test the model on each fold of data
cv_dt <- lapply(folds_dt, function(x) {
        #Creating the training fold without test fold
        training_fold <- train_data[-x,]
        test_fold <- train_data[x,]
        #Building the learner model for each of the 10 unique training fold
        market_dt_kfold <- rpart(purchase~., data=training_fold)
        #Making predictions on each unique test fold
        predict_test_dt_kfold <- predict(market_dt_kfold, newdata = test_fold, type = "class")
        #Creating confusion matrix for each unique test fold
        cm_dt <- table(predict_test_dt_kfold, factor(test_fold$purchase))
        #Calculating accuracy for each fold
        accuracy_dt <- (cm_dt[1,1] + cm_dt[2,2]) / (cm_dt[1,1] + cm_dt[2,2] + cm_dt[1,2] + cm_dt[2,1])
        return(accuracy_dt)
}) 
cv_dt

#Taking the mean of all the 10 accuracy values to determine the ultimate model accuracy
mean(as.numeric(cv_dt))

#------------------------------------------------------------------------------------------------------------

#Applying k-Fold Cross Validation to Naive Bayes Model
#Creating the number of folds
folds_nb <- createFolds(train_data$purchase, k = 10)

#Creating the function to train and test the model on each fold of data
cv_nb <- lapply(folds_nb, function(x) {
        #Creating the training fold without test fold
        training_fold <- train_data[-x,]
        test_fold <- train_data[x,]
        #Building the learner model for each of the 10 unique training fold
        market_nb_kfold <- naiveBayes(purchase~., data=training_fold)
        #Making predictions on each unique test fold
        predict_test_nb_kfold <- predict(market_nb_kfold, newdata = test_fold, type = "class")
        #Creating confusion matrix for each unique test fold
        cm_nb <- table(predict_test_nb_kfold, factor(test_fold$purchase))
        #Calculating accuracy for each fold
        accuracy_nb <- (cm_nb[1,1] + cm_nb[2,2]) / (cm_nb[1,1] + cm_nb[2,2] + cm_nb[1,2] + cm_nb[2,1])
        return(accuracy_nb)
}) 
cv_nb

#Taking the mean of all the 10 accuracy values to determine the ultimate model accuracy
mean(as.numeric(cv_nb))
#---------------------------------------------------------------------------------------------

#Using C5.0 algorithm for decision trees
install.packages("C50")
library('C50')

c_tree <- C5.0(purchase ~.,data=train_data)
c_tree


#Making predictions on the test data
predict_test_c5 <- predict(c_tree, newdata = test_data, type = "class")
predict_test_c5

##Building the confusion matrix
confusionMatrix(predict_test_c5, factor(test_data$purchase))










