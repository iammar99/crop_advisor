# ===============================
#     Installing Libraries
# ===============================

library(caret)
library(nnet)
library(e1071)
library(rpart)
library(rpart.plot)


# ===============================
#     Loading Dataset
# ===============================

data <- read.csv("G:\\ML\\Crop Recomendation system\\Crop_recommendation.csv")
head(data)

# ===============================
# Getting a little bit details
# ===============================


colnames(data)

# "N",  "P",  "K",  "temperature",  "humidity", "ph", "rainfall", "label"

unique(data$label)
data$label <- factor(data$label)

#  [1] "rice"        "maize"       "chickpea"    "kidneybeans" "pigeonpeas"  "mothbeans"   "mungbean"    "blackgram"   "lentil"      "pomegranate"
# [11] "banana"      "mango"       "grapes"      "watermelon"  "muskmelon"   "apple"       "orange"      "papaya"      "coconut"     "cotton"     
# [21] "jute"        "coffee" 




# ===============================
#       Creating sample
# ===============================


set.seed(123)
train_index <- createDataPartition(data$label, p=0.8, list=FALSE)

train_data <- data[train_index,]
test_data <- data[-train_index,]



# ===============================
#   Applying Logistic Regression
# ===============================



model_log <- multinom(label ~ . , data = train_data , maxit = 1000)
prediction_log <- predict(model_log, newdata = test_data)

summary(model_log)

conf_matrix_log <- confusionMatrix(data = prediction_log, reference = test_data$label)






# ===============================
#       Applying KNN
# ===============================


fit_control <- trainControl(method = "cv", number = 10)
model_knn <- train(label ~ ., 
                   data = train_data, 
                   method = "knn", 
                   trControl = fit_control, 
                   preProcess = c("center", "scale"), 
                   tuneLength = 10) # Best value 5
test_predictions_knn <- predict(model_knn, newdata = test_data)
knn_confusion_matrix <- confusionMatrix(test_predictions_knn, test_data$label)
print(knn_confusion_matrix)





# ===============================
#       Applying SVM
# ===============================


# ====== Linear Kernel ====== Best with 1

model_svm_lin <- svm(label ~ . , data = train_data , kernel = "linear",cost = 1)
prediction_svm_lin <- predict(model_svm_lin,newdata = test_data)
conf_table_lin <- confusionMatrix(test_data$label, prediction_svm_lin)
summary(model_svm_lin)



# ====== Radial Kernel ======

model_svm_rad <- svm(label ~ . , data = train_data , kernel = "radial",cost = 1)
prediction_svm_rad <- predict(model_svm_rad,newdata = test_data)
conf_table_rad <- confusionMatrix(test_data$label, prediction_svm_rad)
summary(model_svm_rad)



# ====== Polynomial Kernel ======


model_svm_pol <- svm(label ~ . , data = train_data , kernel = "polynomial")
prediction_svm_pol <- predict(model_svm_pol,newdata = test_data)
conf_table_pol <- confusionMatrix(test_data$label, prediction_svm_pol)
summary(model_svm_pol)


print(conf_table_lin$overall["Accuracy"])
print(conf_table_rad$overall["Accuracy"])
print(conf_table_pol$overall["Accuracy"])




# ===============================
#     Applying Decision Tree
# ===============================


tree_model <- rpart(label ~ ., data = train_data, method = "class")
rpart.plot(tree_model, type = 2, extra = 104, fallen.leaves = TRUE, main = "Crop Recommendation Decision Tree")
tree_predictions <- predict(tree_model, newdata = test_data, type = "class")
conf_matrix_tree <- confusionMatrix(data = tree_predictions, reference = test_data$label)



# ===============================
#     Applying Decision Tree
# ===============================


model_nb <- naiveBayes(label ~ ., data = train_data)
prediction_nb <- predict(model_nb, newdata = test_data)
conf_matrix_nb <- confusionMatrix(data = prediction_nb, reference = test_data$label)





# ===============================
#   Creating Comparison Table
# ===============================

# ------------ Logictic Regression ------------

comp_table <- data.frame(
  Name = "Logistic Regression",
  AIC = 384.3235,
  Kappa = 0.969,
  Accuracy = 97
)

# ------------ KNN ------------


newRow <- list(Name = "KNN", AIC = NA , Kappa = 0.973 ,Accuracy = 97 )
comp_table <- rbind(comp_table,newRow)


# ------------ SVM ------------


newRow <- list(Name = "SVM (Linear)", AIC = NA , Kappa = 0.983 ,Accuracy = 98.4 )
comp_table <- rbind(comp_table,newRow)
newRow <- list(Name = "SVM (Radial)", AIC = NA , Kappa = 0.981 ,Accuracy = 98.1 )
comp_table <- rbind(comp_table,newRow)
newRow <- list(Name = "SVM (Polynomial)", AIC = NA , Kappa = 0.916 ,Accuracy = 92 )
comp_table <- rbind(comp_table,newRow)


# ------------ Decision Tree ------------


newRow <- list(Name = "Decision Tree", AIC = NA , Kappa = 0.95  ,Accuracy = 95.2 )
comp_table <- rbind(comp_table,newRow)


# ------------ Naive Bayes ------------


newRow <- list(Name = "Naive Bayes", AIC = NA , Kappa = 0.995  ,Accuracy = 99.5 )
comp_table <- rbind(comp_table,newRow)













# ===========================| Table |===========================

#         Name                  AIC           Kappa     Accuracy
#   1.    Logistic Regression   384.3235      0.969     97.0
#   2.    KNN                   NA            0.973     97.0
#   3.    SVM (Linear)          NA            0.983     98.4
#   4.    SVM (Radial)          NA            0.981     98.1
#   5.    SVM (Polynomial)      NA            0.916     92.0
#   6.    Decision Tree         NA            0.950     95.2
#   7.    Naive Bayes           NA            0.995     99.5

