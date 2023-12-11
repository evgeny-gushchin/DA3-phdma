### Assignment 3 Prediction with ML for Economists Fall 2023 ###
### by Evgeny Gushchin

# CLEAR MEMORY
rm(list=ls())

#install.packages("margins")
#install.packages("kableExtra") # I get an error
#install.packages("Hmisc")
#install.packages("gmodels")
#install.packages("lspline")
#install.packages("rattle")
#install.packages("partykit")
#install.packages("libcoin")
#install.packages("rpart.plot")

# Import libraries
library(haven)
library(glmnet)
library(purrr)
library(margins)
library(skimr)
#library(kableExtra)
library(Hmisc)
library(cowplot)
library(gmodels) 
library(lspline)
library(sandwich)
library(modelsummary)
library(tidyr)

library(rattle)
library(caret)
library(pROC)
library(ranger)
library(rpart)
library(partykit)
library(rpart.plot)
library(dplyr)

setwd("/Users/evgenygushchin/Documents/GitHub/DA3-phdma/A3")
getwd() 
options(digits = 3)
##### open & filter raw data ######
data_original <- read.csv("cs_bisnode_panel.csv") # this is the latest data available
data_original %>% glimpse()

# Calculate the proportion of NA values in each column
na_proportion <- colMeans(is.na(data_original))
na_proportion <- round(na_proportion, 2)
# Print the result
print(na_proportion)
# many variabes have no NA values - good!
# can get rid of variables that are all NAs: COGS, finished_prod, net_dom_sales, net_exp_sales, wages, D

table(data_original$year)

data <- data_original %>% 
  select(-c(COGS, finished_prod, net_dom_sales, net_exp_sales, wages, D)) %>%
  filter(year <2016 & year >2009) # thus we end up with the sample of 2010-2015
data %>% glimpse() # 167,606 rows and 42 columns

summary(data$exit_year)
length(data$exit_year[is.na(data$exit_year)]) # probably it is better to drop companies that have exited

data <- data %>% filter(is.na(exit_year))
data %>% glimpse() # now we have 150,648 observations

# add all missing year and comp_id combinations -
# originally missing combinations will have NAs in all other columns
data <- data %>%
  complete(year, comp_id)
data %>% glimpse() # now we have 201,816 observations

# creating our target variable (fast growth)
summary(data$sales)
length(data$sales[data$sales < 0]) # some negative numbers, though not too many

summary(data$fixed_assets)
length(data$fixed_assets[data$fixed_assets < 0]) # few negative numbers
length(data$fixed_assets[data$fixed_assets == 0]) # a few zeros

summary(data$inc_bef_tax)
length(data$inc_bef_tax[data$inc_bef_tax < 0]) # a half of observations has negative income

summary(data$share_eq)
length(data$share_eq[data$share_eq < 0]) # almost a half of observations has negative shareholder equity

# Option 1: The fixed asset turnover ratio (FAT)
#FAT is useful in determining whether a company is efficiently using its fixed assets to drive net sales. 
#FAT is calculated by dividing net sales by the average balance of fixed assets of a period. 
#Though the ratio is helpful as a comparative tool over time or against other companies, it fails to identify unprofitable companies.
#https://www.investopedia.com/terms/f/fixed-asset-turnover.asp

data <- data %>%
  mutate(sales = ifelse(sales < 0, 1, sales),
         ln_sales = ifelse(sales > 0, log(sales), 0),
         sales_mil=sales/1000000,
         sales_mil_log = ifelse(sales > 0, log(sales_mil), 0))

data <- data %>%
  mutate(fixed_assets = ifelse(fixed_assets < 0, 1, fixed_assets),
         ln_fixed_assets = ifelse(fixed_assets > 0, log(fixed_assets), 0),
         fixed_assets_mil=fixed_assets/1000000,
         fixed_assets_mil_log = ifelse(fixed_assets > 0, log(fixed_assets_mil), 0))

data <- data %>% mutate(fat = ifelse(fixed_assets != 0, sales/fixed_assets, 0))
summary(data$fat)

data <- data %>%
  mutate(fat = ifelse(fat < 0, 1, fat),
         ln_fat = ifelse(fat > 0, log(fat), 0))

data <- data %>%
  group_by(comp_id) %>%
  mutate(d1_ln_fat = ln_fat - lag(ln_fat, 1) ) %>%
  ungroup()

summary(data$d1_ln_fat)
data %>% filter(!is.na(d1_ln_fat) & d1_ln_fat>5) %>% dim() # lets say that 5% growth rate of FAT is out threshold for fast growing firm

data  <- data %>%
  mutate(fast_growth = ifelse(d1_ln_fat > 5 & !is.na(d1_ln_fat),1,0))
table(data$fast_growth) # we have just 1166 of fast growing firms
table(data$fast_growth, data$year)

data <- data %>% 
  group_by(comp_id) %>%
  mutate(future_fast_growth = lead(fast_growth,1)) %>% 
  ungroup()
table(data$future_fast_growth, data$year)

# Option 3: Growth of number of employees (labor_avg)
# we have too many missing values (~50%)

# Option 4: Growth rate of Shareholder equity (share_eq)
# might have nothing to do with the company's efficiency

# look at cross section
data <- data %>%
  filter(year == 2012) %>%
  # look at firms below 10m euro revenues and above 1000 euros
  filter(!(sales_mil > 10)) %>%
  filter(!(sales_mil < 0.001))
table(data$future_fast_growth) #very unbalanced

# Model 1: Logit

# Model 2: Random Forest

# Model 3: Ensamble of Logit + RF

#=================== Task 1 =====================
#### PART I: Probability prediction ####

#### PART II: Classification ####

#### PART III: Discussion of results ####

#=================== Task 2 =====================
