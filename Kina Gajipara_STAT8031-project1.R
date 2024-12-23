                               # project:1

library(tidyverse)
library(ggplot2)
library(stargazer)
library(caret)
library(leaps)
library(corrplot)

#loading salary dataset
dt <- read.csv('C:/Users/divya/Downloads/Salaries.csv')   

                  ## Data Preproccessing:
# Count missing values in each column
missing_count <- sum(is.na(dt))  
View(missing_count)

# removing column which only shows index
dt <- select(dt, -rownames)           

# checking for the duplicates rows
duplicate_count <- sum(duplicated(dt))
cat("Number of duplicate rows:", duplicate_count, "\n")

# Display duplicate rows (optional)
duplicate_rows <- dt[duplicated(dt), ]
print(duplicate_rows)
                  # it shows 4 duplicate but it's not a duplicates

                  ## Exploratory Data Analysis and Visualization
# summary provide min.,max.mean and quartiles which helps to predict salary in this case
summary(dt)               

# Distribution of salary
dt %>% ggplot(aes(salary)) + geom_histogram()
              #from the graph of salary we can easily see that around 40 people have salary around 100000 and 
              #it is highest in numbers. However,there is only one person who has highest salary around 240000. 

# Factorization of categorical variable into numerical  
dt$rank = as.numeric(as.factor(dt$rank))
dt$discipline = as.numeric(as.factor(dt$discipline))
dt$sex = as.numeric(as.factor(dt$sex))

# visualizing correlation matrix to know correlation between variables
cor_matrix <- cor(dt)

corrplot(cor_matrix, method = "color", type = "full", 
         col = colorRampPalette(c("blue", "white", "red"))(200),
         title = "Correlation Matrix Heatmap",
         cex.main = 1.5,
         addCoef.col = "black")
              # from the correlation we can see that both have positive value it means if years since phd and 
              # experience increase salary is also increase accordingly. we can also check it graphically.

# converting back to categorical
dt$rank <- factor(dt$rank)
dt$sex <- factor(dt$sex)
dt$discipline <- factor(dt$discipline)

dt %>% ggplot(aes(x = rank , y = salary)) + geom_boxplot()
                    # here,rank are categorical variable and we can see that professor earn more compare to associate
                    # professor and assistant professor.

dt %>% ggplot(aes(x = sex , y = salary)) + geom_boxplot()
                    # here, male earn more compare to female 

dt %>% ggplot(aes(x = discipline , y = salary)) + geom_boxplot()
                    # there is grade(A and B) in discipline if people have grade B then they earn more.

                      
                    ## Feature Selection and Model Building

set.seed(123)
train_indices <- createDataPartition(dt$salary, p = 0.7, list = FALSE)
train_data <- dt[train_indices, ]
test_data <- dt[-train_indices, ]

# Perform subset selection to find the best predictors
subsetmodel <- regsubsets(salary ~ rank + discipline + yrs.since.phd + yrs.service + sex, 
                          data = train_data, nbest = 1)  # nbest = 1 returns the best model
summary(subsetmodel)

# Plot the Adjusted R-squared to identify the best model
plot(subsetmodel, scale = "adjr2")  # Adjusted R-squared is a good criterion for feature selection

# Identify which predictors are selected in the best model
best_model_predictors <- summary(subsetmodel)$which[which.max(summary(subsetmodel)$adjr2), ]
best_model_predictors
           # according to the this plot top best model is the large model because all predictors are included

#creating a CV model 
selected_formula <- salary ~ rank + discipline + yrs.since.phd + yrs.service

# Perform Cross-Validation with the selected predictors
set.seed(123)  
cv_model <- train(
  form = selected_formula,
  data = train_data,
  method = "lm",  # Linear regression
  trControl = trainControl(method = "cv", number = 10)  # 10-fold CV
)

# Display CV results
print(cv_model)

# Evaluate model on test data
predictions <- predict(cv_model, newdata = test_data)
actual_values <- test_data$salary

# Calculate RMSE and MAE
rmse <- sqrt(mean((predictions - actual_values)^2))
mae <- mean(abs(predictions - actual_values))

cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")

# Compare predicted vs actual values
plot(predictions, actual_values, xlab = "Predicted Salary", ylab = "Actual Salary", main = "Predicted vs Actual")
abline(0, 1, col = "red")  # reference line

residuals <- actual_values - predictions

# Check Normality: QQ Plot
qqnorm(residuals, main = "QQ Plot of Residuals")
qqline(residuals, col = "red")

# Shapiro-Wilk Test for Normality
shapiro_test <- shapiro.test(residuals)
cat("Shapiro-Wilk Test p-value:", shapiro_test$p.value, "\n")

if (shapiro_test$p.value > 0.05) {
  cat("Residuals are approximately normal.\n")
} else {
  cat("Residuals are not normal. Consider transformations or alternative models.\n")
}

# Check Linearity: Residuals vs. Fitted values
plot(predictions, residuals, 
     xlab = "Predicted Values", ylab = "Residuals", main = "Residuals vs Fitted plot")
abline(h = 0, col = "red", lty = 2)

# Check Normality: Histogram of Residuals
hist(residuals, breaks = 20, main = "Histogram of Residuals", xlab = "Residuals", col = "blue")



