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

# creating our target variable (fast growth)

# Model 1: Logit

# Model 2: Random Forest

# Model 3: Ensamble of Logit + RF

#=================== Task 1 =====================
#### PART I: Probability prediction ####

#### PART II: Classification ####

#### PART III: Discussion of results ####

#=================== Task 2 =====================
