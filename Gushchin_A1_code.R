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
# location is important but we have too many state, we should find a way to group them

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

state_means <- data %>% group_by(stfips) %>% summarise(y_mean = mean(y)) %>% arrange(y_mean)
state_means$state_group <- NA
state_means$state_group[1:17] <- "Low-pay states"
state_means$state_group[18:34] <- "Medium-pay states"
state_means$state_group[35:51] <- "High-pay states"
table(state_means$state_group)
data <- data %>% right_join(state_means, by = "stfips")
table(data$state_group)
# now I try to use the data on state-level minimum wage in 2014
# source: https://www.dol.gov/agencies/whd/state/minimum-wage/history
data <- data %>% mutate(min_wage = ifelse(stfips %in% c("AK", "DE", "NV"), 7.75, 
                                   ifelse(stfips == "AR", 6.25,                                    
                                   ifelse(stfips %in% c("AZ", "MT"), 7.9, 
                                   ifelse(stfips == "CA", 9,
                                   ifelse(stfips %in% c("CO", "MA", "NY", "RI"), 8, 
                                   ifelse(stfips == "CT", 8.7,
                                   ifelse(stfips == "FL", 7.93,
                                   ifelse(stfips %in% c("GA", "WY"), 5.15,
                                   ifelse(stfips %in% c("HI", "ID", "IN", "IA", "KS", "KY", "MD", "MN", "NE", "NH", "ND", "NC", "PA", "SD", "TX", "UT", "VA", "WV", "WI", "AL", "LA", "SC", "MS", "TN"), 7.25, 
                                   ifelse(stfips %in% c("IL", "NJ"), 8.25,
                                   ifelse(stfips %in% c("ME", "MO"), 7.5,
                                   ifelse(stfips == "MI", 8.15,
                                   ifelse(stfips == "NM", 7.5,
                                   ifelse(stfips == "OH", 7.6,
                                   ifelse(stfips == "OK", 4.625,
                                   ifelse(stfips == "OR", 9.1,
                                   ifelse(stfips == "VT", 8.73,
                                   ifelse(stfips == "WA", 9.32,
                                   ifelse(stfips == "DC", 9.5,9999999))))))))))))))))))))
summary(data$min_wage) 
# AL, LA, MS, SC and TN have not adopted a state minimum wage, thus the federal rate of 7.25 is applied

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
# we can reduce the number of groups here
data <- data %>% mutate(race_new = ifelse(race == 1, "White", 
                                   ifelse(race == 2, "Black", 
                                   ifelse(race == 3, "American Indian", 
                                   ifelse(race == 4, "Asian", "Other" )))))
table(data$race_new)

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
# we can make this variable in 5 groups
data <- data %>% mutate(industry = ifelse(ind02 == "Drinking places, alcoholic beverages (7224)", "Drinking places, alcoholic beverages (7224)", 
                                   ifelse(ind02 == "Other amusement, gambling, and recreation industries (713 exc. 71395)", "Other amusement, gambling, and recreation industries (713 exc. 71395)", 
                                   ifelse(ind02 == "Restaurants and other food services (722 exc. 7224)", "Restaurants and other food services (722 exc. 7224)", 
                                   ifelse(ind02 == "Traveler accommodation (7211)", "Traveler accommodation (7211)", "Other" )))))
table(data$industry)
# $occ2012 - SOC-based occupation code (used for filtering)
# $class - Class of worker
table(data$class) # mostly Private, For Profit 
# we can unite all governement
data <- data %>% mutate(class_new = ifelse(class == "Private, For Profit", "Private, For Profit", 
                                    ifelse(class == "Private, Nonprofit", "Private, Nonprofit", "Government" )))
table(data$class_new)

# $unionmme - Union member (1 - Yes, 2 - No)
table(data$unionmme) # almost all are not members
# $unioncov - Covered by a union contract (1 - Yes, 2 - No)
table(data$unioncov)# mostly No
table(data$unioncov,data$unionmme)
#we can make one new variable from these two
data <- data %>% mutate(union_status = ifelse(unionmme == "Yes", "Union member/contract",
                                       ifelse(unioncov == "Yes", "Union member/contract", "Non-member/no contract")))

table(data$union_status)
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

g3 <- ggplot(data, aes(x = factor(state_group), y = y,
                       )) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "State",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g3
# $stfips does seem to matter

g3a <- ggplot(data, aes(x = factor(min_wage), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "State",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g3a 
#
g4 <- ggplot(data, aes(x = factor(intmonth), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Month",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g4
# doesn't seem that $intmonth matters

g5 <- ggplot(data, aes(x = factor(race_new), y = y,
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

g10 <- ggplot(data, aes(x = factor(industry), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Industry",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g10
# doesn't seem that $ind02 matters somewhat

g11 <- ggplot(data, aes(x = factor(class_new), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Class",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g11
# seems that $class matters 


g12 <- ggplot(data, aes(x = factor(union_status), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Union status",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g12
# seems that $union_status matters 

g13 <- ggplot(data, aes(x = factor(lfsr94), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Employment status",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g13
# doesn't seem that $lfsr94 matters 
g14 <- ggplot(data, aes(x = factor(sex), y = y,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Sex",y = "Wage (USD per hour)")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 40), breaks = seq(0,40, 10))+
  theme_minimal()
g14

# List of column names you want to convert to factors
columns_to_factor <- c("educ_level", "class_new", "union_status","lfsr94", "prcitshp", "marital", "sex", "race_new", "state_group","intmonth", "grade92", "ownchild", "chldpres", "industry")

# Use lapply to apply as.factor to the specified columns
data[columns_to_factor] <- lapply(data[columns_to_factor], as.factor)

data <- data %>% mutate(age_squared = age^2)

# models 1-4
# Model 1: Linear regression on age
model1 <- as.formula(y ~ age + age_squared + educ_level)
model2 <- as.formula(y ~ age + age_squared + educ_level + min_wage + union_status)
model3 <- as.formula(y ~ age + age_squared + educ_level + min_wage + union_status + sex + class_new + age:sex)
model4 <- as.formula(y ~ age + age_squared + educ_level + min_wage + union_status + sex + class_new + age:sex + race_new + prcitshp + industry + educ_level:sex)

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


