### Assignment 1 Prediction with ML for Economists Fall 2023 ###
rm(list = ls()) # cleaning the environment

library(ggplot2)
library(dplyr)
#library(tidyverse)

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
# hhid - Household id
length(unique(data$hhid)) # we have 2055 unique HH

# models 1-4

# calculate RSME

# clculate cross-validated RMSE

# calculate BIC

# visualize the results

