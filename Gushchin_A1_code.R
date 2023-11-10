### Assignment 1 Prediction with ML for Economists Fall 2023 ###
rm(list = ls()) # cleaning the environment

#install.packages("fixest")
#install.packages("modelsummary")

library(ggplot2)
library(dplyr)
library(fixest)
library(lmtest)
library(sandwich)
library(modelsummary)
library(tidyverse)
library(stargazer)
library(caret)
library(grid)

# open data
data_original <- read.csv("morg-2014-emp.csv")
data_original %>% glimpse()

# filter and process data
data <- data_original %>% filter(occ2012 == 4110) 
# we are keeping only the occupation of our interest Waiters and waitresses
data %>% glimpse()
data %>% dim() # we end up with 2178 observations

# now let's figure out what data do we have
length(unique(data$X)) # all unique -> X must be the ID variable
# $hhid - Household id
length(unique(data$hhid)) # we have 2055 unique HH
# $intmonth - Interview calendar month
table(data$intmonth) # the respondents are more or less evenly distributed across months
# can we expect seasonality in the wages reported? Possibly
table(data$stfips)
# $stfips - state of the respondent
# $weight - Final Weight x 100
# $earnwke - Earnings per week
hist(data$earnwke)
summary(data$earnwke) # no zeroes
# $uhours - Usual hours
hist(data$uhours)
summary(data$uhours) # no zeroes
data$y = data$earnwke/data$uhours # this is our variable of interest
data <- data %>% mutate(y = earnwke/uhours)
hist(data$y)
summary(data$y) 
# $grade92 - highest educational grade completed
table(data$grade92)
# most cases are High school graduate, diploma or GED (code 39)
# or Some college but no degree (code 40)
# we can transform this variable to reduce its size
data <- data %>% mutate(educ_level = ifelse(grade92 < 39, "No high school diploma", 
                                            ifelse(grade92 < 41, "High school diploma", 
                                            ifelse(grade92 == 41, "Associate degree -- occupational/vocational", 
                                            ifelse(grade92 == 42, "Associate degree -- academic program", 
                                            ifelse(grade92 == 43, "Bachelor's degree",
                                            ifelse(grade92 < 46, "Master's degree/Professional school", "PhD" )))))))

table(data$educ_level)
# $race - race
table(data$race) # most cases 1 - White, 2 - Black and 4 - Asian
table(data$ethnic) # ‘What is the origin or descent of ...?’ 
#This variable subdivides the Hispanic community by national origin of ancestry.
# $age - age
hist(data$age)
summary(data$age) 
# $sex - sex (1 - Male, 2 - Female)
table(data$sex) # ~75% in the sample are women 
# $marital - marital status (1-3 - married, 4 - widowed, 5 - divorced, 
# 6 - separated, 7 - never married)
table(data$marital) # mostly never married
# $ownchild - Number of own children less than 18 in primary family.
table(data$ownchild) # mostly no children
# $chldpres - Presence of own children less than 18 in primary family.
table(data$chldpres)
# $prcitshp - Citizenship status.
table(data$prcitshp) # mostly Native, Born In US
# $state - 1960 Census Code for state.
table(data$state) # sometimes digits, sometimes letters (messy)
# $ind02 - 3-digit NAICS-based industry code
table(data$ind02) # mostly Restaurants and other food services (722 exc. 7224) 
# $occ2012 - SOC-based occupation code (used for filtering)
# $class - Class of worker
table(data$class) # mostly Private, For Profit 
# $unionmme - Union member (1 - Yes, 2 - No)
table(data$unionmme) # almost all are not members
# $unioncov - Covered by a union contract (1 - Yes, 2 - No)
table(data$unioncov) # mostly No
# $lfsr94 - 
table(data$lfsr94) # all employed, mostly at work

# Calculate the proportion of NA values in each column
na_proportion <- colMeans(is.na(data))
# I asked Chat GPT for help here. Link:
# https://chat.openai.com/share/08b11841-6020-4f5b-80cc-3c9e3ebb52f8

# Print the result
print(na_proportion)
# ethnic is the only variable that has NA values (>80%)

#visualizing

g1 <- ggplot(data, aes(x=age, y=y), fill = factor(unionmme), color=factor(unionmme)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Age",y = "Wage (USD per hour)")+
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = FALSE, , alpha = 0.1, linetype = "dashed")
g1

g2 <- ggplot(data, aes(x = factor(educ_level), y = y,
                        fill = factor(sex), color=factor(sex))) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  #stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  #scale_color_manual(name="",
  #                   values=c(color[2],color[1])) +
  #scale_fill_manual(name="",
  #                  values=c(color[2],color[1])) +
  labs(x = "Education Level",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal() +
  theme(legend.position = c(0.3,0.8)        )
g2
#save_fig("A1-figure2-wages_education_sex", output, "small")

g3 <- ggplot(data, aes(x = factor(stfips), y = y,
                       )) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "State",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g3
# $stfips does seem to matter
#save_fig("A1-figure2-wages_education_sex", output, "small")

g4 <- ggplot(data, aes(x = factor(intmonth), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Month",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g4
# doesn't seem that $intmonth matters

g5 <- ggplot(data, aes(x = factor(race), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Race",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g5
# doesn't seem that $race matters much (5 is more likely a dummy for Hawaii)

g6 <- ggplot(data, aes(x = factor(marital), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Marital status",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g6
# doesn't seem that $marital matters much

g7 <- ggplot(data, aes(x = factor(ownchild), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Number of children",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g7
# doesn't seem that $ownchild matters 

g8 <- ggplot(data, aes(x = factor(chldpres), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Presence of children",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g8
# doesn't seem that $chldpres matters much

g9 <- ggplot(data, aes(x = factor(prcitshp), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Citizenship",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g9
# doesn't seem that $prcitshp matters much

g10 <- ggplot(data, aes(x = factor(ind02), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Industry",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g10
# doesn't seem that $ind02 matters somewhat

g11 <- ggplot(data, aes(x = factor(class), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Class",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g11
# seems that $class matters 

g12 <- ggplot(data, aes(x = factor(unionmme), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Union membership",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g12
# seems that $unionmme matters 

g13 <- ggplot(data, aes(x = factor(unioncov), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Union contract",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g13
# seems that $unioncov matters 

g14 <- ggplot(data, aes(x = factor(lfsr94), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Employment status",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g14
# doesn't seem that $lfsr94 matters 
g15 <- ggplot(data, aes(x = factor(sex), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Sex",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g15

# List of column names you want to convert to factors
columns_to_factor <- c("educ_level", "class", "unionmme", "unioncov","lfsr94", "prcitshp", "marital", "sex", "race", "stfips","intmonth", "grade92", "ownchild", "chldpres", "ind02")

# Use lapply to apply as.factor to the specified columns
data[columns_to_factor] <- lapply(data[columns_to_factor], as.factor)

data <- data %>% mutate(age_squared = age^2)

# models 1-4
# Model 1: Linear regression on age
model1 <- as.formula(y ~ age + educ_level)
model2 <- as.formula(y ~ age + educ_level + age_squared + sex)
model3 <- as.formula(y ~ age + educ_level + age_squared + sex + unionmme + marital + class)
model4 <- as.formula(y ~ age + educ_level + age_squared + sex + marital + race + class + unionmme + prcitshp + ind02 + stfips)

# Running simple OLS
reg1 <- feols(model1, data=data, vcov = 'hetero')
reg2 <- feols(model2, data=data, vcov = 'hetero')
reg3 <- feols(model3, data=data, vcov = 'hetero')
reg4 <- feols(model4, data=data, vcov = 'hetero')
# calculate RSME

models <- c("reg1", "reg2","reg3", "reg4")
AIC <- c()
BIC <- c()
RMSE <- c()
RSquared <- c()
regr <- c()
regressors <- c()
# Get for all models
for ( i in 1:length(models)){
  AIC[i] <- AIC(get(models[i]))
  BIC[i] <- BIC(get(models[i]))
  RMSE[i] <- sqrt(sum((data$y-predict(get(models[i])))^2)/length(data$y))
  regr[[i]] <- coeftest(get(models[i]), vcov = sandwich)
  regressors[i] <- length(get(models[i])$coefficients)-1
}

#################################################################
# Cross-validation

# set number of folds
k <- 4

set.seed(13505)
cv1 <- train(model1, data, method = "lm", trControl = trainControl(method = "cv", number = k))
set.seed(13505)
cv2 <- train(model2, data, method = "lm", trControl = trainControl(method = "cv", number = k))
set.seed(13505)
cv3 <- train(model3, data, method = "lm", trControl = trainControl(method = "cv", number = k), na.action = "na.omit")
set.seed(13505)
cv4 <- train(model4, data, method = "lm", trControl = trainControl(method = "cv", number = k), na.action = "na.omit")

# calculate average rmse
cv <- c("cv1", "cv2", "cv3", "cv4")
rmse_cv <- c()

for(i in 1:length(cv)){
  rmse_cv[i] <- sqrt((get(cv[i])$resample[[1]][1]^2 +
                        get(cv[i])$resample[[1]][2]^2 +
                        get(cv[i])$resample[[1]][3]^2 +
                        get(cv[i])$resample[[1]][4]^2)/4)
}


# summarize results
cv_mat <- data.frame(rbind(cv1$resample[4], "Average"),
                     rbind(cv1$resample[1], rmse_cv[1]),
                     rbind(cv2$resample[1], rmse_cv[2]),
                     rbind(cv3$resample[1], rmse_cv[3]),
                     rbind(cv4$resample[1], rmse_cv[4])
)

colnames(cv_mat)<-c("Resample","Model1", "Model2", "Model3", "Model4")
cv_mat

# all results in one table
main_table <- data.frame(rbind(t(BIC), t(RMSE), rmse_cv, regressors))
colnames(main_table)<-c("Model1", "Model2", "Model3", "Model4")
row.names(main_table) <- c("BIC", "RMSE (full sample)", "RMSE (4-fold cross validation)", "Number of regressors")
main_table

stargazer(main_table, summary = F, digits=2, float=F, out="A1-results-table-Gushchin.tex")
stargazer(main_table, summary = F, digits=2, float=F, type="text",  out="A1-results-table-Gushchin.tex")


