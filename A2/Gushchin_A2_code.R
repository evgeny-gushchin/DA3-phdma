### Assignment 2 Prediction with ML for Economists Fall 2023 ###
### by Evgeny Gushchin
rm(list = ls()) # cleaning the environment

library(ggplot2)
library(dplyr)

setwd("/Users/evgenygushchin/Documents/GitHub/DA3-phdma/A2")
getwd() 

# open & clean raw data
data_original <- read.csv("listings_Vienna_09_2023.csv") # this is the latest data available
data_original %>% glimpse()

# Calculate the proportion of NA values in each column
na_proportion <- colMeans(is.na(data_original))
na_proportion <- round(na_proportion, 2)
# Print the result
print(na_proportion)
# many variabes have no NA values - good!
# can get rid of variables that are all NAs: neighbourhood_group_cleansed, bathrooms, calendar_updated, license

data <- data_original %>% select(-c(neighbourhood_group_cleansed, bathrooms, calendar_updated, license))
data %>% glimpse()

summary(data$accommodates) 
data <- data %>% filter(accommodates<7 & accommodates>1)
summary(data$accommodates) # now we have the data for places that can accomodate 2-6 people

data <- data %>% filter(room_type=="Entire home/apt") # because we are interested only in apartments
data %>% glimpse() # now we are down to 10,409 observations

table(data$property_type)
data <- data %>% filter(property_type=="Entire loft" | property_type=="Entire serviced apartment" | property_type=="Entire rental unit" | property_type=="Entire place") # these seem filter out non-apartments
data %>% glimpse() # now we are down to 9,233 observations

print(data$price[1]) # we need to clean this variable
data$price_cleaned <- gsub("[^0-9.]", "", data$price)
# Convert to numeric, handle NAs
data$price_numeric <- as.numeric(data$price_cleaned)
print(data$price_numeric[1]) # done!

summary(data$price_numeric)
