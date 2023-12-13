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
library(ggplot2)

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

# change some industry category codes
table(data$ind2)
data <- data %>%
  mutate(ind2_cat = ind2 %>%
           ifelse(. > 56, 60, .)  %>%
           ifelse(. < 26, 20, .) %>%
           ifelse(. < 55 & . > 35, 40, .) %>%
           ifelse(. == 31, 30, .) %>%
           ifelse(is.na(.), 99, .)
  )

table(data$ind2_cat)

summary(data$d1_ln_fat)

# replace w 0 for new firms + add dummy to capture it
data <- data %>%
  mutate(age = (year - founded_year) %>%
           ifelse(. < 0, 0, .),
         new = as.numeric(age <= 1) %>% #  (age could be 0,1 )
           ifelse(balsheet_notfullyear == 1, 1, .),
         d1_ln_fat = ifelse(new == 1, 0, d1_ln_fat),
         new = ifelse(is.na(d1_ln_fat), 1, new),
         d1_ln_fat = ifelse(is.na(d1_ln_fat), 0, d1_ln_fat))

# Firm characteristics
summary(data$age)
table(data$foreign)
table(data$gender)
table(data$region_m)
data <- data %>%
  mutate(age2 = age^2,
         foreign_management = as.numeric(foreign >= 0.5),
         gender_m = factor(gender, levels = c("female", "male", "mix")),
         m_region_loc = factor(region_m, levels = c("Central", "East", "West")))

# assets can't be negative. Change them to 0 and add a flag.
data <-data  %>%
  mutate(flag_asset_problem=ifelse(intang_assets<0 | curr_assets<0 | fixed_assets<0,1,0  ))
table(data$flag_asset_problem)

data <- data %>%
  mutate(intang_assets = ifelse(intang_assets < 0, 0, intang_assets),
         curr_assets = ifelse(curr_assets < 0, 0, curr_assets),
         fixed_assets = ifelse(fixed_assets < 0, 0, fixed_assets))

# generate total assets
data <- data %>%
  mutate(total_assets_bs = intang_assets + curr_assets + fixed_assets)
summary(data$total_assets_bs)


pl_names <- c("extra_exp","extra_inc",  "extra_profit_loss", "inc_bef_tax" ,"inventories",
              "material_exp", "profit_loss_year", "personnel_exp")
bs_names <- c("intang_assets", "curr_liab", "fixed_assets", "liq_assets", "curr_assets",
              "share_eq", "subscribed_cap", "tang_assets" )

# divide all pl_names elements by sales and create new column for it
data <- data %>%
  mutate_at(vars(pl_names), funs("pl"=./sales))

# divide all bs_names elements by total_assets_bs and create new column for it
data <- data %>%
  mutate_at(vars(bs_names), funs("bs"=ifelse(total_assets_bs == 0, 0, ./total_assets_bs)))


########################################################################
# creating flags, and winsorizing tails
########################################################################

# Variables that represent accounting items that cannot be negative (e.g. materials)
zero <-  c("extra_exp_pl", "extra_inc_pl", "inventories_pl", "material_exp_pl", "personnel_exp_pl",
           "curr_liab_bs", "fixed_assets_bs", "liq_assets_bs", "curr_assets_bs", "subscribed_cap_bs",
           "intang_assets_bs")

data <- data %>%
  mutate_at(vars(zero), funs("flag_high"= as.numeric(.> 1))) %>%
  mutate_at(vars(zero), funs(ifelse(.> 1, 1, .))) %>%
  mutate_at(vars(zero), funs("flag_error"= as.numeric(.< 0))) %>%
  mutate_at(vars(zero), funs(ifelse(.< 0, 0, .)))


# for vars that could be any, but are mostly between -1 and 1
any <-  c("extra_profit_loss_pl", "inc_bef_tax_pl", "profit_loss_year_pl", "share_eq_bs")

data <- data %>%
  mutate_at(vars(any), funs("flag_low"= as.numeric(.< -1))) %>%
  mutate_at(vars(any), funs(ifelse(.< -1, -1, .))) %>%
  mutate_at(vars(any), funs("flag_high"= as.numeric(.> 1))) %>%
  mutate_at(vars(any), funs(ifelse(.> 1, 1, .))) %>%
  mutate_at(vars(any), funs("flag_zero"= as.numeric(.== 0))) %>%
  mutate_at(vars(any), funs("quad"= .^2))


# dropping flags with no variation
variances<- data %>%
  select(contains("flag")) %>%
  apply(2, var, na.rm = TRUE) == 0

data <- data %>%
  select(-one_of(names(variances)[variances]))

########################################################################
# additional
# including some imputation
########################################################################

# CEO age
data <- data %>%
  mutate(ceo_age = year-birth_year,
         flag_low_ceo_age = as.numeric(ceo_age < 25 & !is.na(ceo_age)),
         flag_high_ceo_age = as.numeric(ceo_age > 75 & !is.na(ceo_age)),
         flag_miss_ceo_age = as.numeric(is.na(ceo_age)))

data <- data %>%
  mutate(ceo_age = ifelse(ceo_age < 25, 25, ceo_age) %>%
           ifelse(. > 75, 75, .) %>%
           ifelse(is.na(.), mean(., na.rm = TRUE), .),
         ceo_young = as.numeric(ceo_age < 40))

# number emp, very noisy measure
data <- data %>%
  mutate(labor_avg_mod = ifelse(is.na(labor_avg), mean(labor_avg, na.rm = TRUE), labor_avg),
         flag_miss_labor_avg = as.numeric(is.na(labor_avg)))

summary(data$labor_avg)
summary(data$labor_avg_mod)

data <- data %>%
  select(-labor_avg)

# create factors
data <- data %>%
  mutate(urban_m = factor(urban_m, levels = c(1,2,3)),
         ind2_cat = factor(ind2_cat, levels = sort(unique(data$ind2_cat))))

data <- data %>%
  mutate(future_fast_growth_f = factor(future_fast_growth, levels = c(0,1)) %>%
           recode(., `0` = 'no_fast_growth', `1` = "fast_growth"))

########################################################################
# sales 
########################################################################

data <- data %>%
  mutate(sales_mil_log_sq=sales_mil_log^2)


ggplot(data = data, aes(x=sales_mil_log, y=as.numeric(future_fast_growth))) +
  geom_point(size=2,  shape=20, stroke=2, fill="blue", color="blue") +
  geom_smooth(method = "lm", formula = y ~ poly(x,2), color="green", se = F, size=1)+
  geom_smooth(method="loess", se=F, colour="yellow", size=1.5, span=0.9) +
  labs(x = "sales_mil_log",y = "future_fast_growth") +
  theme_minimal()


# Model 1: Logit

# Model 2: Random Forest

# Model 3: Ensamble of Logit + RF

#=================== Task 1 =====================
#### PART I: Probability prediction ####

#### PART II: Classification ####

#### PART III: Discussion of results ####

#=================== Task 2 =====================
