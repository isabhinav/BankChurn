# Exploring and Predicting Customer Churn 
#### _Leveraging Data to Identify At-Risk Customers and Reduce Churn in Banking_

## Introduction
Customer churn is a significant concern for businesses since acquiring new customers is often more expensive than retaining existing ones. Losing customers can also lead to a decline in revenue and profitability. Additionally, high churn rates can be an indicator of deeper issues with customer satisfaction and loyalty. Therefore, identifying and retaining at-risk customers is critical to maintaining a successful business.  

In this project, we use exploratory data analysis techniques to uncover potential risk factors for churn and predict customer churn using machine learning classification algorithms in R. By doing so; we aim to provide insights to inform effective customer retention strategies at the bank.

## Problem Statement
A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them which customers they can expect to churn so they can proactively intervene and offer services and products to retain the customer, and ultimately achieve a positive return on investment for the bank.

##  Business Problem Perspective
In designing a solution for the churn classification problem, it is essential to consider various factors in the decision-making process. These factors can include but are not limited to 
- the resources available to the bank manager
- the bank's business priorities
- marketing cost

The selection of appropriate evaluation metrics hinges upon a comprehensive understanding of these factors and their influence on the business objectives. 

For example, If the manager emphasizes capturing as many potential churners as possible, the model should prioritize maximizing recall. This means the model aims to identify a high proportion of customers who are likely to churn, even if it results in some false positives. On the other hand, if the bank manager has limited resources and wants to focus on targeting only the most likely churners, the model should prioritize precision. This means the model aims to minimize false positives and identify the customers most likely to churn, even if the model may miss some potential churners. 

Moreover, for churn problems, it is essential to consider the impact on lifetime value when assessing the effectiveness of the models. However, calculating lifetime value involves making numerous assumptions about the business such as the method of outreach (direct mail or email), gross margin, cost of outreach, nature of incentive (offer or call to action), cost of the offer, open rate of direct mail or email, and response rate to these marketing interventions. Given the complexity of these assumptions, we need a less comprehensive metric, which still helps assess our models' financial impact.

By carefully considering these elements, we can tailor our approach and metrics to effectively address the churn prediction problem and align with the specific needs and goals of the bank.


## Data Description
- Dataset obtained from Kaggle: https://www.kaggle.com/sakshigoyal7/credit-card-customers
- The dataset contains information about credit card customers, including demographic information, credit card usage, and relationship with the bank.
- The data is at Client ID level and consists of 10,127 rows and 23 columns.
- Includes both numeric and categorical variables, and there are no missing values in the dataset.
- Also includes a binary target variable, indicating whether the customer has churned or not.
- Approximately 16.1% of customers have churned, indicating an imbalanced class problem - where the occurrence of churned customers is relatively low compared to non-churned customers.


##  Exploratory Data Analysis
Focused on correlation plots, density plots, boxplots for numerical variables, and the distribution of categorical variables. Below are a few highlights from the analysis.

##### _Correlation Plots_
 - Correlation analysis identified key variables for churn prediction
 - Correlation plots revealed the most highly correlated numeric variables with the response variable.
 - These correlations suggest that the transaction behavior and their level of engagement with the bank are both factors that are important in predicting churn.
 - Among the most highly correlated variables with the response variable are 'Total_Trans_Ct',' Total_Ct_Chng_Q4_Q1', and 'Total_Revolving_Bal'.
 - Other highly correlated columns, such as 'Credit_Limit' and 'Avg_Open_To_Buy', must be evaluated and handled before data modeling.

 
![corrplot](https://github.com/isabhinav/BankChurn/assets/130937665/424bd0d4-c7b5-4f36-a424-6f9205d8a8fb)

##### _Density Plots_
 - Density plots highlighted behavioral differences between customer groups.
 - Density plots demonstrated differences in behavior between churned and existing customers, particularly in transaction variables such as 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Revolving_Bal', and 'Avg_Utilization_Ratio'.
 - For example, we observe that the highest density of 'Total_Trans_Ct' for churned customers is around 80, while for existing customers, it is closer to 50.
 - Some differences were observed in customer relationship variables as well.


 ![eda_num1](https://github.com/isabhinav/BankChurn/assets/130937665/5ee97b4b-ca4e-48f6-8ed9-7657834e4161)

##### _Boxplots_
 - Boxplots facilitated the detection of distribution disparities and outliers.
 - Boxplots helped highlight distribution differences between churned and existing customers and potential outliers in the data.
 - Notably, transaction variables such as 'Total_Revolving_Bal' and 'Avg_Utilization_Ratio' had median values close to zero for churned customers.
- The plots helped identify outliers; we must handle these data points appropriately to ensure model performance.


 ![eda_num2](https://github.com/isabhinav/BankChurn/assets/130937665/be5c5070-8468-4562-84f0-60b1bdcca2dc)

 ##### _Bar Charts_
 - Bar Charts revealed demographic patterns among churned and non-churned customers.
 - Examined categorical variables' distributions using bar charts.
 - Single customers appeared more likely to churn than married customers, and the income category seems to influence customer status.
- We observe rows marked as 'Unknown' for 'Marital_Status','Education_Level', and 'Income_Category'.


![eda_cat1](https://github.com/isabhinav/BankChurn/assets/130937665/d9eb6374-b3cd-47d3-97cc-83fdf3afcffc)

## Data Preparation
 - Columns deemed irrelevant for the analysis were dropped from the dataset.
- Rows containing 'Unknown' values in certain features such as marital_status, income_category, and education_level were kept in the dataset. This decision was based on the assumption that customers intentionally chose not to disclose this information. Retaining these rows allows capturing this behavior as a potential signal for customer churn.
- Highly correlated variables were identified and dropped from the dataset. In particular, 'Avg_Open_To_Buy' was removed since it had a perfect correlation with 'Credit_Limit' and offered no significant business logic for the modeling process.
- Variables were converted to appropriate factor and numeric types. 
- Categorical variables were transformed into dummy variables for modeling purposes.
- Numeric data were scaled using robust scaling. Robust scaling reduces the impact of outliers on the scaling process, making it suitable for datasets containing outliers while preserving the relationships between data points.
- The dataset was split into 75% training data and 25% testing data for model training and evaluation.

## Model Selection
 Selected the following models to predict churn:
- Stepwise logistic regression - is interpretable and allows for variable selection. 
- Random forest -  excels at handling outliers and capturing complex nonlinear relationships between predictors and response variable.
- XGBoost - combines ensemble learning and gradient boosting to achieve high predictive performance.


## Evaluation Metrics
Traditionally, classification models in churn analysis are evaluated using metrics such as accuracy and Kappa. While these metrics provide a general assessment of model performance, they may not directly align with the specific objectives of the business and account for the consequences of misclassification in our churn problem. We take a more tailored approach to evaluate our models to address this.

Our evaluation strategy focuses on metrics that directly quantify the consequences of correct and incorrect predictions, considering the unequal cost associated with misclassification in our churn problem. In particular, we prioritize two key metrics: recall and profit per send.

##### _Recall_
Recall, also known as sensitivity or true positive rate, measures the ability of a model to correctly identify the positive class (churners) out of all the actual positive instances. In our context, recall is a crucial metric because we assume that the bank wants to capture as many potential churners as possible, even if it means intervening with some customers who may not churn. 

##### _Expected Value_
Suppose we know the costs and benefits of each type of classification (true positives, true negatives, false positives, false negatives). Incorporating these costs into the evaluation of classification models makes it possible to estimate the financial impact of each model. Although our expected value calculation is not comprehensive, it still sheds light on the business impact of various models and the costs associated with misclassification. 

##### _Additional Metrics_
We also explore additional metrics to gain a comprehensive understanding of the trade-offs between different models. By considering these additional metrics, we can gain deeper insights into the performance of our models and make informed decisions about the optimal model selection based on the specific priorities and requirements of the business. For instance, we can examine the F1 score, which balances both precision and recall, providing insight into the overall effectiveness of the model in capturing churners while minimizing false positives. Additionally, we can plot the ROC curve to visualize the discrimination ability of our models and analyze the trade-off between the true positive rate and the false positive rate.

##### _Summary_
Our evaluation approach goes beyond traditional metrics and focuses on metrics that align with the specific objectives of the business. By assessing recall and expected value as primary metrics, along with exploring other relevant metrics such as the F1 score and ROC curve, we ensure a comprehensive understanding of the model's performance and the associated trade-offs.

## Model Evaluation
- In evaluating our models, we considered several metrics along with their 95% confidence intervals: AUC, F1 score, recall, and expected value. The results are compared below
<img width="664" alt="compare_metrics" src="https://github.com/isabhinav/BankChurn/assets/130937665/ddd618de-c26e-4528-8f6e-7494c901b986">


- Although there may be some overlap in the confidence intervals, the consistent trends observed across the evaluation metrics suggest that the XGBoost model generally outperforms the stepwise logistic and random forest models in terms of recall, F1 score, AUC, and profit per send.

- While the confidence intervals provide a measure of uncertainty, the overall pattern of superior performance in the XGBoost model still indicates its strengths in churn classification.

![roc_comparison](https://github.com/isabhinav/BankChurn/assets/130937665/f0b8d6ee-a171-469a-8f4c-062fe238e89e)

- From the ROC curves, we see that the XGBoost model performs the best in terms of managing both specificity and sensitivity.

##### _Lifetime and Expected Value_

<img width="355" alt="ltv_churners" src="https://github.com/isabhinav/BankChurn/assets/130937665/146f67c3-ab63-48c3-b7ec-690569fca459">
 
- Lifetime Value of Churners: 
   - Avg annual transaction amount for churners is $3,095
   - Avg lifetime period for churners is 3 years (35.8 months)    
   - Margin per transaction is 3% (assumption)
   - LTV (lifetime value of a customer) = total annual transaction amount * margin per transaction * retention time period 
   - Therefore, the average LTV for a churner is 3095 * 0.03 * 3 = $278.5
     

- Expected Value of Model: 
   - Marketing cost is $100 (assumption)
   - when a customer accurately predicted to churn responds to the intervention (TP), the profit is the difference between the customer's Lifetime Value ($278.5) and the marketing cost ($100).
   - if the model incorrectly predicts that a customer will churn (FP), the only loss is the cost of the promotion ($100). 
   - if a customer who is about to churn is inaccurately predicted not to churn (FN), the potential Lifetime Value (278.5) is lost.
   - Expected Value of Model = (278.5 - 100) * tp - 100 * fp - 278.5 * fn

- In summary, we can conclude that out of the models we evaluated, the XGBoost model is the most promising and impactful for our churn classification problem.
  
