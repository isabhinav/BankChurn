# Load csv
BankChurn <- read.csv(file.choose())
mydata <- BankChurn


# Load necessary packages
library(tidyverse) # data manipulation and visualization in R
library(corrplot) # create correlation matrix plots in R
library(pheatmap) # create heatmap plots in R
library(caret) # train and evaluate predictive models in R
library(glmnet) # fit lasso and regularized linear models in R
library(rattle) # data mining and machine learning in R
library(ROCR) # visualize the performance of binary classifiers in R
library(pROC) # computing and visualizing ROC curves and related metrics in R
library(randomForest) # fit random forest models in R
library(flextable) # creating tables in markdown documents
library(gtsummary) # creating summary tables of regression models in R
library(summarytools) # creating summary statistics tables in R
library(MASS) # functions for applied statistics 
library(gridExtra) # arranging multiple plots in a grid in R
library(psych) # descriptive statistics and related methods in R
library(htmltools) # work with html code in R
library(scales) # format axis labels and scales in ggplot2 plots
library(plotly) # create interactive plots in R
library(RColorBrewer) # create color palettes in R
library(car) # applied regression analysis in R
library(e1071) # statistical learning and data mining in R
library(gridExtra) # arrange multiple grid-based plots on a page
library(xgboost) # implementation of  gradient boosting algorithm called XGBoost
library(dplyr)
library(boot)
library(knitr)
library(formattable)

# Inspect data
head(mydata,5)


# Drop useless columns
mydata <- mydata %>% 
  dplyr::select(-c(Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1, 
                   Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2,
                   CLIENTNUM))


# Describe data 
view(dfSummary(mydata, plain.ascii = FALSE, style = "grid", tmp.img.dir = "/tmp"))

# Dataset has 10,127 observations and 21 columns.
# Target variable ("Attrition_Flag") is a binary variable, indicating whether a customer has churned or not.
# Dataset has both numeric and categorical variables.
# There are no missing values in the dataset.
# The data describes customer behavior from the past 12 months.


# Exploratory data analysis
# EDA - numerical variables - correlation matrix
mydata <- mydata %>%
  mutate(Attrition_Flag = ifelse(Attrition_Flag == "Attrited Customer", "1", "0"))

mydata$Attrition_Flag <- as.numeric(mydata$Attrition_Flag) # convert to numeric to calculate correlation
num_mydata <- mydata %>% select_if(is.numeric) # subset numerical values for correlation plot

cor_mat <- cor(num_mydata) # create correlation matrix
corrplot(cor_mat, method="color", # examine correlations
         type="upper", tl.cex = 0.8, 
         col = colorRampPalette(c("darkblue", "lightblue", "white", "orange", "red"))
         (100), diag = FALSE, addCoef.col = "black", number.cex = 0.7, tl.srt = 90)

# The most highly correlated variables with the response are 
# - 'Total_Trans_Ct' (|r| = 0.37)
# - 'Total_Ct_Chng_Q4_Q1' (|r| = 0.29)
# - 'Total_Revolving_Bal' (|r| = 0.26)
# - 'Avg_Utilization_Ratio'(|r| = 0.18)
# - 'Total_Relationship_Count' and 'Months_Inactive_12_mon' (|r| = 0.15)
# These insights suggest that the transaction behavior and their level of engagement with the bank are both factors that are important in predicting churn. 
# Other highly correlated columns such as 'Credit_Limit' and 'Avg_Open_To_Buy' need to be evaluated and handled before data modeling


# EDA - numerical variables - density plots
# Exploring whether the correlated variables listed above behave differently for attrited and existing customers
# Define variables and corresponding correlation coefficients

vars <- c('Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Revolving_Bal', 'Avg_Utilization_Ratio', 'Total_Relationship_Count', 'Months_Inactive_12_mon')
## cors <- c(-0.37, -0.29, -0.26, -0.18, -0.15, 0.15)

# Create a list of plots
plot_list <- list()

# Loop over variables and create density plot for each
for (i in 1:length(vars)) {
  p <- ggplot(mydata, aes(x = .data[[vars[i]]], fill = as.factor(Attrition_Flag))) +
    geom_density(alpha = 0.5) +
    labs(y = "Density",
         title = paste("Density plot of", vars[i])) +
    scale_fill_brewer(palette = "Set1") +
    theme_minimal() +
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5),
          panel.border = element_rect(color = "black", fill = NA, size = 0.12))
  
  plot_list[[i]] <- p
}

# Arrange density plots in a grid
grid.arrange(grobs = plot_list, ncol = 2)

# density plot observations
# Churned and existing customers behave differently for transaction variables such as 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Revolving_Bal', and 'Avg_Utilization_Ratio'
# For example, we see observe that the density of' Total_Trans_Ct' for churned customers is around 80, while for existing customers, it is closer to 50
# There seem to be slight differences in the behavior of churned and existing customers for relationship variables such as 'Total_Relationship_Count' and 'Months_Inactive_12_mon' as well
# Combining these graphs with boxplots can help get a complete picture of the distribution of the variables and their relationship with churn


# EDA - numerical variables - boxplots
# Create box plot plot for each variable
for (i in 1:length(vars)) {
  p <- ggplot(mydata, aes(x = .data[[vars[i]]], fill = as.factor(Attrition_Flag))) +
    geom_boxplot(alpha = 0.5, outlier.alpha = 0.6, width = 0.1) +
    labs(y = "Density",
         title = paste("Density plot of", vars[i])) +
    scale_fill_brewer(palette = "Set1") +
    theme_minimal() +
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5),
          panel.border = element_rect(color = "black", fill = NA, size = 0.12))
  
  plot_list[[i]] <- p
}

# Arrange box plots in a grid
grid.arrange(grobs = plot_list, ncol = 2)

# boxplot observations
# The plots highlight the difference in the distribution of the variables for existing and churned customer as well as the outlier data points
# Note that the median value of 'Total_Revolving_Bal' and 'Avg_Utilization_Ratio' is nearly zero for the churned customers
# We also observe that the spread of the data can be different for the two groups
# There are instances of outlier data points across our variables and these outliers can potentially influence our model performance
# Outliers need to evaluated and handled accordingly



# EDA -  categorical variables
# Examine distribution of categorical variables for churned and existing customers
plot_list <- list()

for (var in c("Gender", "Education_Level", "Marital_Status", "Income_Category")) 
{
  
  p <- ggplot(mydata, aes(x = as.factor(Attrition_Flag), y = ..count.., fill = .data[[var]])) +
    geom_bar(position = "fill", stat = "count", alpha = 0.75, color= 'black') +
    labs(x = "Attrition Flag", y = "Count", fill = var, 
         title = paste("Distribution by Attrition_Flag and", var)) +
    scale_y_continuous(labels = percent) +
    scale_fill_brewer(palette = "Set1") +
    theme_minimal() +
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5),
          panel.border = element_rect(color = "black", fill = NA, size = 0.12))
  
  plot_list[[var]] <- p
}

grid.arrange(grobs = plot_list, ncol = 2)

# categorical variables observations
# Proportion of female customers is higher among the churned customers as compared to the non-churned customers.
# There is no observable pattern in the proportion of churned vs. not churned customers across different education levels.
# Single customers seem to be more likely to churn compared to married customers.
# Income category seems to have some effect on customer status
# We observe that there are rows marked as 'Unknown' for 'Marital_Status','Education_Level', and 'Income_Category'



# Data preparation
# We chose to keep rows that had 'Unknown' values for certain features such as marital_status, income_category, and education_level. 
# This decision was based on the assumption that these values may represent instances where customers intentionally choose not to disclose this information. 
# By keeping these rows in the data set, we are able to capture this behavior and potentially use it as a signal for customer churn. 
# For example, customers who choose not to disclose their income_category may be more likely to churn, and this information can be valuable for predicting and preventing churn. 
# Also, removing these rows can lead to potentially introducing bias

# Data backup for modeling
pre_model_data <- mydata

# Convert response to factor variable
pre_model_data$Attrition_Flag <- as.factor(pre_model_data$Attrition_Flag)

# Drop highly correlated variables
# Out of all the highly correlated columns, 'Avg_Open_To_Buy' has the least significance to the modeling process
# It has a perfect correlation with 'Credit_Limit' and does not offer any significant business logic to keep in our modeling dataset
pre_model_data <- pre_model_data %>% select(-Avg_Open_To_Buy)


# select categorical variables to convert to dummies
vars_char <- c("Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category")
vars_num <- c("Customer_Age", 
              "Dependent_count", 
              "Months_on_book", 
              "Total_Relationship_Count", 
              "Months_Inactive_12_mon", 
              "Contacts_Count_12_mon", 
              "Credit_Limit", 
              "Total_Revolving_Bal", 
              "Total_Amt_Chng_Q4_Q1", 
              "Total_Trans_Amt", 
              "Total_Trans_Ct", 
              "Total_Ct_Chng_Q4_Q1", 
              "Avg_Utilization_Ratio")
Attrition_Flag <- pre_model_data$Attrition_Flag


# create dummy variables
dummies <- dummyVars(formula = ~., data = pre_model_data[,vars_char])

# apply dummies to the data
model_dummies <- data.frame(predict(dummies, newdata = pre_model_data))

# Scale numeric data
# Robust scaling is a method that aims to reduce the impact of outliers on the scaling process
# It is a good choice for datasets that contain outliers and where we want to maintain the relationships between data points without skewing the distribution
# model_numeric_scaled <- model_numeric %>%
#  mutate_if(is.numeric, list(~ (.-median(.))/mad(.)))

model_numeric <- pre_model_data[vars_num]
model_numeric_scaled <- model_numeric %>%
  mutate_if(is.numeric, list(~ (.-median(.))/mad(.)))

# combine scaled numerical, dummy categorical and response variables into one dataset
# combine numerical and dummy variables
# model_data <- cbind(model_numeric_scaled, model_dummies,Attrition_Flag)

# add Attrition_Flag column back after scaling and combining with dummy variables
model_data <- cbind(model_numeric_scaled, model_dummies)
model_data <- cbind(model_data, Attrition_Flag)

# check the structure of the new data frame
str(model_data)


# split data into test and training sets
set.seed(123)

# Split data into 75% training and 25% testing
trainIndex <- createDataPartition(model_data$Attrition_Flag, p = 0.75, list = FALSE)
train_data <- model_data[trainIndex, ]
test_data <- model_data[-trainIndex, ]

# Create train control for models
# train control 1 - cv with 10 fold
fitControl <- trainControl(method = "cv", number = 10, savePredictions = TRUE)


# Modeling
# Model 1 - Stepwise Logistic regression (Baseline model)
log_model <- glm(Attrition_Flag ~ ., data = train_data, family = binomial)
summary(log_model)

## NAs in the output and hence doesnt make prediction. need another baseline model
## or remove some of the variables and run the code again

# Logistic regression with stepwise
step_model <- stepAIC(log_model,
                      direction = "both",
                      trace = F,
                      trControl = fitControl)

# Create object of importance of our variables
step_importance <- varImp(step_model) %>%
  rownames_to_column("Variable") %>%
  arrange(desc(Overall))

# Create plot of importance of variables
ggplot(data = step_importance, aes(x = Overall, y = reorder(Variable, Overall))) +
  geom_bar(stat = "identity") +
  labs(title = "Variable Importance: Stepwise Logistic Regression Model",
       x ="Importance", y = "Feature") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(color = "black", fill = NA, size = 0.12))

# Make predictions using the stepwise logistic regression model on the test dataset
step_prediction <- predict(step_model, newdata = test_data, type = "response")

# Convert the predicted probabilities to a data frame and add a column for the predicted class (1 if probability is greater than 0.5, 0 otherwise)
step_prediction <- step_prediction %>%
  as.data.frame() %>%
  magrittr::set_colnames(c("prob")) %>%
  mutate(pred_class = ifelse(prob > 0.5, 1, 0))

# Create a table of predicted class vs. actual class and calculate proportions
table(step_prediction$pred_class, test_data$Attrition_Flag) %>% 
  prop.table() %>% 
  round(2) 

# Show the confusion matrix for the predictions 
confusionMatrix(table(step_prediction$pred_class, test_data$Attrition_Flag), positive = "1")

# data check
# F1 score = 2 * (precision * recall) / (precision + recall)
# f1_step = 2 * (0.76375 * 0.58128) / (0.76375 + 0.58128)
# 0.6601378


## evaluation metrics
## calculate confidence intervals around metrics 
compute_metrics <- function(step_prediction, true_labels) {
  cm <- confusionMatrix(step_prediction, reference = true_labels, positive = "1")
  recall <- cm$byClass["Sensitivity"]
  f1 <- cm$byClass["F1"]
  return(c(recall, f1))
}

# Convert variables to factors
prediction_factors <- as.factor(step_prediction$pred_class)
label_factors <- as.factor(test_data$Attrition_Flag)

# Define the bootstrap function
bootstrap_func <- function(data, indices) {
  preds <- data$step_prediction[indices]
  labels <- data$true_labels[indices]
  compute_metrics(preds, labels)
}

# Prepare the data for bootstrapping
bootstrap_data <- data.frame(step_prediction = prediction_factors,
                             true_labels = label_factors)

# Set the number of bootstrap iterations
n_iterations <- 1000

# Perform bootstrapping
boot_results <- boot(data = bootstrap_data, statistic = bootstrap_func, R = n_iterations)

# Calculate confidence intervals
step_ci_recall <- boot.ci(boot_results, type = "basic", index = 1)
# 95%   ( 0.5360,  0.6297 )  
step_ci_f1 <- boot.ci(boot_results, type = "basic", index = 2)
# 95%   ( 0.6219,  0.7017 )  



# Plot ROC and Calculate AUC 
# Calculate predicted probabilities for the positive class
step_prob <- predict(step_model, newdata = test_data, type = "response")

# Create ROC curve
step_roc_obj <- roc(test_data$Attrition_Flag, step_prob)

# Plot ROC curve
plot(step_roc_obj, main = "ROC Curve: Stepwise Logistic Regression Model", print.auc = TRUE)
# auc - 0.930

# Calculate AUC with 95% confidence interval
step_auc_obj <- ci.auc(step_roc_obj, conf.level = 0.95)
# 95% CI: 0.9178-0.9424 (DeLong)

# profit per send
# 26.4 * tp - 2 * fp - 28.4 * fn
# step_profit = 26.4 * 236 - 2 * 170 - 28.4 * 73
# 3817.2


# Model 2 - Random Forest 
rf_model <- train(
  Attrition_Flag ~ .,  # Set Y variable followed by "~." to include all variables in formula.
  method = 'rf',  # Set method as random forest.
  trControl = fitControl,  # Set cross validation settings
  data = train_data)  # Set data as train_data.

# Create object of importance of our variables
rf_importance <- varImp(rf_model)

# Create box plot of importance of variables
ggplot(data = rf_importance, mapping = aes(x = rf_importance1[,1])) + 
  geom_boxplot() + 
  labs(title = "Var Importance: Random Forest Model") + 
  theme_light() + 
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold")) +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(color = "black", fill = NA, size = 0.12))

# Predict using the Random Forest model
rf_prediction <- predict(rf_model, test_data, type = "prob")

# Assign predicted class based on a threshold of 0.5
rf_prediction <- rf_prediction %>%
  mutate(pred_class = ifelse(rf_prediction[, 2] > 0.5, 1, 0))

# Create a table of predicted class vs. actual class and calculate proportions
prediction_table <- table(rf_prediction$pred_class, test_data$Attrition_Flag) %>%
  prop.table() %>%
  round(2)

# Generate a confusion matrix using the predicted class and actual class, with "1" as the positive class
confusionMatrix(table(rf_prediction$pred_class, test_data$Attrition_Flag), positive = "1")

# data check
# f1_rf = 2 * (0.9110 * 0.8571) / (0.9110 + 0.8571)
# 0.8832284



## evaluation metrics
## calculate confidence intervals around metrics 
compute_metrics <- function(rf_prediction, true_labels) {
  cm <- confusionMatrix(rf_prediction, reference = true_labels, positive = "1")
  recall <- cm$byClass["Sensitivity"]
  f1 <- cm$byClass["F1"]
  return(c(recall, f1))
}

# Convert variables to factors
prediction_factors <- as.factor(rf_prediction$pred_class)
label_factors <- as.factor(test_data$Attrition_Flag)

# Define the bootstrap function
bootstrap_func <- function(data, indices) {
  preds <- data$rf_prediction[indices]
  labels <- data$true_labels[indices]
  compute_metrics(preds, labels)
}

# Prepare the data for bootstrapping
bootstrap_data <- data.frame(rf_prediction = prediction_factors,
                             true_labels = label_factors)

# Set the number of bootstrap iterations
n_iterations <- 1000

# Perform bootstrapping
boot_results <- boot(data = bootstrap_data, statistic = bootstrap_func, R = n_iterations)

# Calculate confidence intervals
rf_ci_recall <- boot.ci(boot_results, type = "basic", index = 1)
# 95%   ( 0.8257,  0.8928 )  
rf_ci_f1 <- boot.ci(boot_results, type = "basic", index = 2)
# 95%   ( 0.8606,  0.9074 )  


# Calculate predicted probabilities for the positive class
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, 2]

# Create ROC curve
rf_roc_obj <- roc(test_data$Attrition_Flag, rf_prob)

# Plot ROC curve
plot(rf_roc_obj, main = "ROC Curve: Random Forest Model", print.auc = TRUE)
# auc 0.988

# Calculate AUC with 95% confidence interval
rf_auc_obj <- ci.auc(rf_roc_obj, conf.level = 0.95)
# 95% CI: 0.9843-0.9917 (DeLong)


# profit per send
# 26.4 * tp - 2 * fp - 28.4 * fn
# rf_profit = 26.4 * 348 - 2 * 58 - 28.4 * 34
# 8105.6


#### Model 3 - xgboost 
xgb_model <- train(
  Attrition_Flag ~ ., 
  method = 'xgbTree',  # Set method as xgboost
  trControl = fitControl,
  data = train_data
)

# Create object of importance of our variables
xgb_importance <- varImp(xgb_model)

# Create box plot of importance of variables
ggplot(data = xgb_importance, mapping = aes(x = xgb_importance[,1])) +
  geom_boxplot() +
  labs(title = "Variable Importance: XGBoost Model") +
  theme_light() +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold")) +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(color = "black", fill = NA, size = 0.12))

# Predict using the xgboost model on the test_data
xgb_prediction <- predict(xgb_model, test_data, type = "prob")

# Add a new column pred_class based on the predicted probabilities
xgb_prediction <- xgb_prediction %>%
  mutate(pred_class = ifelse(xgb_prediction[,2] > 0.5, 1, 0))

# Create a table of predicted classes versus actual classes and calculate the proportions
table(xgb_prediction$pred_class, test_data$Attrition_Flag) %>%
  prop.table() %>%
  round(2)

# Create a confusion matrix and specify the positive class as "1"
confusionMatrix(table(xgb_prediction$pred_class, test_data$Attrition_Flag), positive = "1")

# data check
# f1_xgb = 2 * (0.9304 * 0.8892) / (0.9304 + 0.8892)
# 0.9152068



## evaluation metrics
## calculate confidence intervals around metrics 
compute_metrics <- function(xgb_prediction, true_labels) {
  cm <- confusionMatrix(xgb_prediction, reference = true_labels, positive = "1")
  recall <- cm$byClass["Sensitivity"]
  f1 <- cm$byClass["F1"]
  return(c(recall, f1))
}

# Convert variables to factors
prediction_factors <- as.factor(xgb_prediction$pred_class)
label_factors <- as.factor(test_data$Attrition_Flag)

# Define the bootstrap function
bootstrap_func <- function(data, indices) {
  preds <- data$xgb_prediction[indices]
  labels <- data$true_labels[indices]
  compute_metrics(preds, labels)
}

# Prepare the data for bootstrapping
bootstrap_data <- data.frame(xgb_prediction = prediction_factors,
                             true_labels = label_factors)

# Set the number of bootstrap iterations
n_iterations <- 1000

# Perform bootstrapping
boot_results <- boot(data = bootstrap_data, statistic = bootstrap_func, R = n_iterations)

# Calculate confidence intervals
xgb_ci_recall <- boot.ci(boot_results, type = "basic", index = 1)
# 95%   ( 0.8621,  0.9208 )  
xgb_ci_f1 <- boot.ci(boot_results, type = "basic", index = 2)
# 95%   ( 0.8907,  0.9303 )   


# Calculate predicted probabilities for the positive class
xgb_prob <- predict(xgb_model, newdata = test_data, type = "prob")[, 2]

# Create ROC curve
xgb_roc_obj <- roc(test_data$Attrition_Flag, xgb_prob)

# Plot ROC curve
plot(xgb_roc_obj, main = "ROC Curve: XGBoost Model", print.auc = TRUE)
# auc 0.992

# Calculate AUC with 95% confidence interval
xgb_auc_obj <- ci.auc(xgb_roc_obj, conf.level = 0.95)
# 95% CI: 0.9885-0.9951 (DeLong)


# profit per send
# 26.4 * tp - 2 * fp - 28.4 * fn
# xgb_profit = 26.4 * 361 - 2 * 45 - 28.4 * 27
# 8673.6



# summary table - ltv calculation 
# Create a new data frame with churned customers (1)
churned_df <- mydata %>%
  filter(Attrition_Flag == 1) %>%
  summarise(total_trans_amnt = mean(Total_Trans_Amt),
            months_on_book = mean(Months_on_book))

# Create a new data frame with not churned customers (0)
not_churned_df <- mydata %>%
  filter(Attrition_Flag == 0) %>%
  summarise(total_trans_amnt = mean(Total_Trans_Amt),
            months_on_book = mean(Months_on_book))

# Combine the two data frames into a single table
summary_table <- rbind(churned_df, not_churned_df)
rownames(summary_table) <- c("Churned (1)", "Not Churned (0)")

# Print the summary table
print(summary_table)

# avg annual transaction amount for churners is $3095
# avg lifetime period for churners is 3 years (35.8 months)
# margin per transaction is 3% (assumption)
# therefore, the average LTV for a churner is 3095 * 0.03 * 3 = $278.5
# marketing cost is $100 (assumption)
# when a customer accurately predicted to churn responds to the intervention (TP), the profit is the difference between the customer's Lifetime Value (278.5) and the marketing cost (100).
# if the model incorrectly predicts that a customer will churn (FP), the only loss is the cost of the promotion (100). 
# if a customer who is about to churn is inaccurately predicted not to churn (FN), the potential Lifetime Value (278.5) is lost.
# ltv_per_send = (278.5 - 100) * tp - 100 * fp - 278.5 * fn 

# ltv per send
# step_ltv = (278.5- 100) * tp - 100 * fp - 278.5 * fn 
step_ev = (178.5) * 236  - 100 * 170 - 278.5 * 73 
# 4795.5

rf_ev = (178.5) * 348 - 100 * 58 - 278.5 * 34 
# 46849

xgb_ev = (178.5) * 361 - 100 * 45 - 278.5 * 27 
# 52419


# compare model metrics
# Create a data frame with the model metrics
model_metrics <- data.frame(
  Model = c("Stepwise Logistic", "Random Forest", "XGBoost"),
  Recall = c("0.5360 - 0.6297", "0.8257 - 0.8928", "0.8621 - 0.9208"),
  F1 = c("0.6219 - 0.7017", "0.8606 - 0.9074", "0.8907 - 0.9303"),
  AUC = c("0.9178-0.9424", "0.9843-0.9917", "0.9885-0.9951"),
  Expected_value = c(4795.5, 46849, 52419)
)

# Format profit values with commas and dollar sign
model_metrics$Expected_value <- currency(model_metrics$Expected_value, symbol = "$", digits = 0, big.mark = ",")

# Create a formatted table
formatted_table <- kable(model_metrics, align = "c", format = "markdown", digits = 2)

# Print the formatted table
print(formatted_table)


# roc curves comparison
# Plot the first ROC curve
plot(step_roc_obj, main = "ROC Curves Comparison", print.auc = FALSE, col = "red")

# Add the second and third ROC curves
lines(rf_roc_obj, col = "blue")
lines(xgb_roc_obj, col = "green")

# Add a legend
legend("bottomright", legend = c("Stepwise Logistic", "Random Forest", "XGBoost"), col = c("red", "blue", "green"), lty = 1, lwd = 2)

# Add labels and gridlines
# abline(0, 1, lty = 2)
# grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
