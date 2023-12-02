### Assignment 2 Prediction with ML for Economists Fall 2023 ###
### by Evgeny Gushchin
rm(list = ls()) # cleaning the environment
#install.packages("installr")
#installr::updateR()
#install.packages("remotes")
#install.packages("skimr")
#install.packages("directlabels")
#install.packages("cowplot")
#remotes::install_github("cran/glmnet")
#install.packages("glmnet", type = "binary")
library(ggplot2)
library(dplyr)
library(stringr)
library(caret)
library(skimr)
library(grid)
library(glmnet)
library(stargazer)
library(xtable)
library(directlabels)
library(knitr)
library(cowplot)
library(tibble)
library(ranger)

setwd("/Users/evgenygushchin/Documents/GitHub/DA3-phdma/A2")
getwd() 
options(digits = 3)
#=================== Task 1 =====================
##### open & filter raw data ######
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

##### data processing & feature engineering ######
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
# we need to treat the outliers 
lower_limit <- 30
upper_limit <- 400

sum(data$price_numeric > upper_limit)
sum(data$price_numeric < lower_limit)
(sum(data$price_numeric > upper_limit) + sum(data$price_numeric < lower_limit))/dim(data)[1]
#So there are 1.75% of observations that are less than 30 dollars per day and more than 400 dollars.
# lets windsorize these observations and create dummies
data$outlier_high = 0
data$outlier_low = 0
data$outlier_low[data$price_numeric < lower_limit] = 1
data$outlier_high[data$price_numeric > upper_limit] = 1
table(data$outlier_high)
data$price_numeric[data$price_numeric < lower_limit] = lower_limit
data$price_numeric[data$price_numeric > upper_limit] = upper_limit

# some columns we can drop because they don't carry any important information for our task
# listing_url, scrape_id, picture_url, host_url, host_picture_url, host_thumbnail_url
# Variables $name, $description, $neighborhood_overview, $host_about require text analysis
# will not use them because of the time constraint
#data <- data %>% select(-c(listing_url, scrape_id, picture_url, host_url, host_picture_url, host_thumbnail_url, price, id, last_scraped, name, description, neighborhood_overview, host_about, host_id, host_name, host_since, host_location, ))
data <- data %>% select(c(outlier_low, outlier_high, price_numeric, neighbourhood_cleansed,property_type,accommodates, bathrooms_text,bedrooms, beds, amenities)) 
data %>% glimpse() 

table(data$neighbourhood_cleansed) # this one is much cleaner than neighbourhood, better to use it
#clean the titles of the districts
data$neighbourhood_cleansed[data$neighbourhood_cleansed == "Rudolfsheim-F\u009fnfhaus"] = "Rudolfsheim-Fünfhaus"
data$neighbourhood_cleansed[data$neighbourhood_cleansed == "W\u008ahring"] = "Währing"
data$neighbourhood_cleansed[data$neighbourhood_cleansed == "Landstra§e"] = "Landstraße"
data$neighbourhood_cleansed[data$neighbourhood_cleansed == "D\u009abling"] = "Döbling"
g1 <- ggplot(data, aes(x = factor(neighbourhood_cleansed), y = price_numeric)) +
  geom_boxplot(alpha = 0.8, na.rm = TRUE, outlier.shape = NA, width = 0.8, fill = "lightblue") +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm = TRUE) +
  labs(x = "Neighbourhood", y = "Apartment price") +
  scale_y_continuous(expand = c(0.01, 0.01), limits = c(0, 200), breaks = seq(0, 200, 20)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
g1
ggsave("Neighbourhood and price.png", g1, width = 8, height = 6, dpi = 300)
# neighbourhood_cleansed is an important variable but we need to find ways to unite some groups or to make it continuous (mean income level?)

# I don't think I need latitude and longitude as I already have neighbourhoods and I don't do the mapping
#data <- data %>% select(-c(latitude, longitude))
#data %>% glimpse()

#table(data$room_type) # this variable was used for filtering

table(data$property_type) # also was used for filtering but there are several groups left, that mey be helpful for prediction
g2 <- ggplot(data, aes(x = factor(property_type), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8,fill = "lightgreen") +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Property type",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g2
ggsave("Property type and price.png", g2, width = 8, height = 6, dpi = 300)

# property_type seems relevant

summary(data$accommodates)
g3 <- ggplot(data, aes(x = factor(accommodates), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8, fill = "orange") +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "How many people can live",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g3
ggsave("How many people can live and price.png", g3, width = 8, height = 6, dpi = 300)

# accommodates is a very important variable (a proxy for apartment size)

table(data$bathrooms_text)
# bathrooms_text also important (can try making it continuous)
data <- data %>% mutate(bathroom = ifelse(bathrooms_text == "0 baths", 0, 
                                   ifelse(bathrooms_text == "Half-bath", 0.5, 
                                   ifelse(bathrooms_text == "1 bath", 1, 
                                   ifelse(bathrooms_text == "1.5 baths", 1.5, 
                                   ifelse(bathrooms_text == "2 baths", 2,
                                   ifelse(bathrooms_text == "2.5 baths", 2.5,
                                   ifelse(bathrooms_text == "3 baths", 3,
                                   ifelse(bathrooms_text == "3.5 baths", 3.5,
                                   ifelse(bathrooms_text == "4 baths", 4,
                                   ifelse(bathrooms_text == "4.5 baths", 4.5,5 )))))))))))
table(data$bathroom) #will use this variable
g4 <- ggplot(data, aes(x = factor(bathroom), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8, fill = "yellow") +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "How many bathrooms",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g4
ggsave("Bathrooms and price.png", g4, width = 8, height = 6, dpi = 300)

table(data$bedrooms) 
# lets regard NAs as zeros
data$bedrooms_flag <- 0
data$bedrooms_flag[is.na(data$bedrooms)] <- 1 #creating a flag variable
data$bedrooms[is.na(data$bedrooms)] <- 0
# having more than 4 bedrooms doesn't make sense because we have apartments that accomodate up to 6 people
# lets put a flag a change these values
#data$bedrooms_outlier_flag = 0
#data$bedrooms_outlier_flag[data$bedrooms >4] == 1
#data$bedrooms[data$bedrooms >4] = 4
g5 <- ggplot(data, aes(x = factor(bedrooms), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8, fill = "grey") +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Bedrooms",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g5
ggsave("Bedrooms and price.png", g5, width = 8, height = 6, dpi = 300)

# bedrooms seems important (continuous)

table(data$beds) 
# again lets regard NAs as zeros
data$beds_flag <- 0
data$beds_flag[is.na(data$beds)] <- 1 #creating a flag variable
data$beds[is.na(data$beds)] <- 0
# again more than 6 beds is strange
# lets put a flag a change these values to 6
#data$beds_outlier_flag = 0
#data$beds_outlier_flag[data$beds >6] == 1
#data$beds[data$beds >6] = 6
g6 <- ggplot(data, aes(x = factor(beds), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8, fill = "pink") +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Beds",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g6
ggsave("Beds and price.png", g6, width = 8, height = 6, dpi = 300)

print(data$amenities[1]) 
# the variable includes a list of amenities, need text analysis
# some amenities are seasonally important (aircon in summer and heating in winter)
# some are individually important (e.g. some people always prefer to have a dishwasher, or a coffee machine)

# also the total size of amenity list can be an important variable
data$TV <- ifelse(grepl("TV", data$amenities, ignore.case = TRUE), 1, 0)
table(data$TV) 

data$wifi <- ifelse(grepl("Wifi", data$amenities, ignore.case = TRUE), 1, 0)
table(data$wifi) # very few don't have wifi

data$dishwasher <- ifelse(grepl("Dishwasher", data$amenities, ignore.case = TRUE), 1, 0)
table(data$dishwasher) 

data$coffeemaker <- ifelse(grepl("Coffee maker", data$amenities, ignore.case = TRUE), 1, 0)
table(data$coffeemaker) 

data$aircon <- ifelse(grepl("Air condition", data$amenities, ignore.case = TRUE), 1, 0)
table(data$aircon) 

data$microwave <- ifelse(grepl("Microwave", data$amenities, ignore.case = TRUE), 1, 0)
table(data$microwave) 

data$heating <- ifelse(grepl("heating", data$amenities, ignore.case = TRUE), 1, 0)
table(data$heating) 

data$fridge <- ifelse(grepl("Refrigerator", data$amenities, ignore.case = TRUE), 1, 0)
table(data$fridge) 

data$bathtub <- ifelse(grepl("Bathtub", data$amenities, ignore.case = TRUE), 1, 0)
table(data$bathtub) 

data$kitchen <- ifelse(grepl("kitchen", data$amenities, ignore.case = TRUE), 1, 0)
table(data$kitchen) 

data$pool <- ifelse(grepl("pool", data$amenities, ignore.case = TRUE), 1, 0)
table(data$pool) 

#help of Chat GPT#
# Create new variable amenity_length
data$amenity_length <- str_count(data$amenities, ",")
# Since the count gives the number of commas, add 1 to get the total number of amenities
data$amenity_length <- data$amenity_length + 1

summary(data$amenity_length) 

g7 <- ggplot(data, aes(x = amenity_length, y = price_numeric, color = factor(accommodates))) + 
  geom_point(alpha = 0.1) +
  theme_minimal() +
  labs(x = "Number of amenities", y = "Apartment price", color = "Accommodates") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 1), se = FALSE, alpha = 0.1, linetype = "dashed")
g7
ggsave("Number of amenities and price by accommodates.png", g7, width = 8, height = 6, dpi = 300)

# amenity_length seems to be important (continuous) 

# List of column names you want to convert to factors
columns_to_factor <- c("neighbourhood_cleansed", "property_type", "wifi", "TV", "dishwasher", "coffeemaker", "aircon", "microwave", "heating", "fridge", "bathtub", "kitchen", "pool", "bedrooms_flag", "beds_flag", "outlier_high", "outlier_low") #"bedrooms_outlier_flag", "beds_outlier_flag",

# Use lapply to apply as.factor to the specified columns
data[columns_to_factor] <- lapply(data[columns_to_factor], as.factor)
data %>% glimpse()
data <- data %>% mutate(accommodates_sq = accommodates^2, bathroom_sq = bathroom^2, beds_sq = beds^2, bedrooms_sq = bedrooms^2, amenity_length_sq = amenity_length^2) 
data <- data %>% mutate(accommodates_cub = accommodates^3, bathroom_cub = bathroom^3, beds_cub = beds^3, bedrooms_cub = bedrooms^3, amenity_length_cub = amenity_length^3) 
data <- data %>% mutate(accommodates_log = log(accommodates), amenity_length_log = log(amenity_length)) 

continuous_vars <- c("accommodates","accommodates_sq", "accommodates_cub", "accommodates_log","bathroom","bathroom_sq", "bathroom_cub", "beds","beds_sq", "beds_cub", "bedrooms","bedrooms_sq", "bedrooms_cub", "amenity_length", "amenity_length_sq", "amenity_length_cub",  "amenity_length_log") 
#Look up property type interactions

# Create a scatter plot with color-coded points
g8<-ggplot(data, aes(x = as.factor(accommodates), y = price_numeric, color = property_type)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5, outlier.shape = NA) +
  labs(x = "Accommodates",
       y = "Price Numeric") +
  theme_minimal()
g8
ggsave("Accommodates and price by property type.png", g8, width = 8, height = 6, dpi = 300)

g9 <- ggplot(data, aes(x = as.factor(accommodates), y = price_numeric, color = wifi)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5, outlier.shape = NA) +
  labs(x = "Accommodates",
       y = "Price Numeric") +
  theme_minimal()
g9
ggsave("Accommodates and price by wifi.png", g9, width = 8, height = 6, dpi = 300)

g10 <- ggplot(data, aes(x = property_type, y = price_numeric, color = wifi)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5, outlier.shape = NA) +
  labs(x = "Property type",
       y = "Price Numeric") +
  theme_minimal()
g10
ggsave("Property type and price by wifi.png", g10, width = 8, height = 6, dpi = 300)

# dummies suggested by graphs
X1  <- c("accommodates*property_type")
X2  <- c("accommodates*wifi", "property_type*wifi")

#################################
# Separate hold-out set #
#################################

# create a holdout set (20% of observations)
smp_size <- floor(0.2 * nrow(data))

# Set the random number generator: It will make results reproducable
#set.seed(20180123)

# create ids:
# 1) seq_len: generate regular sequences
# 2) sample: select random rows from a table
holdout_ids <- sample(seq_len(nrow(data)), size = smp_size)
data$holdout <- 0
data$holdout[holdout_ids] <- 1

#Hold-out set Set
data_holdout <- data %>% filter(holdout == 1)

#Working data set
data_work <- data %>% filter(holdout == 0)

#### First model: OLS + Lasso #####
vars_model_with_inter <- c("price_numeric", continuous_vars, columns_to_factor, X1, X2)
vars_model_without_inter <- c("price_numeric", continuous_vars, columns_to_factor)

# Set lasso tuning parameters
n_folds=5
train_control <- trainControl(method = "cv", number = n_folds)
tune_grid <- expand.grid("alpha" = c(1), "lambda" = seq(0.05, 1, by = 0.05))

formula <- formula(paste0("price_numeric ~ ", paste(setdiff(vars_model_with_inter, "price_numeric"), collapse = " + ")))
formula_alt <- formula(paste0("price_numeric ~ ", paste(setdiff(vars_model_without_inter, "price_numeric"), collapse = " + ")))

set.seed(1234)
lasso_model <- caret::train(formula,
                            data = data_work,
                            method = "glmnet",
                            preProcess = c("center", "scale"),
                            trControl = train_control,
                            tuneGrid = tune_grid,
                            na.action=na.exclude)

print(lasso_model$bestTune$lambda)

lasso_coeffs <- coef(lasso_model$finalModel, lasso_model$bestTune$lambda) %>%
  as.matrix() %>%
  as.data.frame() %>%
  rownames_to_column(var = "variable") %>%
  rename(coefficient = `s1`)  # the column has a name "1", to be renamed

print(lasso_coeffs)

lasso_coeffs_nz<-lasso_coeffs %>%
  filter(coefficient!=0)
print(nrow(lasso_coeffs_nz))

# Evaluate model. CV error:
lasso_cv_rmse <- lasso_model$results %>%
  filter(lambda == lasso_model$bestTune$lambda) %>%
  dplyr::select(RMSE)
print(lasso_cv_rmse[1, 1])

# Print or check the predictions for the holdout
data_holdout$predictions <- predict(lasso_model, data_holdout)
# look at holdout RMSE
model1_rmse <- sqrt(sum((data_holdout$predictions-data_holdout$price_numeric)^2))
model1_rmse
# strange it is too large (it seems we are overfitting)

g11 <- ggplot(data_holdout, aes(x = predictions, y = price_numeric)) + 
  geom_point(alpha = 0.1) +
  theme_minimal() +
  labs(x = "Predictions", y = "Apartment price")+
  geom_smooth(method = "lm", formula = y ~ poly(x, 1), se = FALSE, alpha = 0.1, linetype = "dashed")
g11
ggsave("Prediction errors.png", g11, width = 8, height = 6, dpi = 300)

#### Second model: Random Forest #####

# do 5-fold CV
#train_control <- trainControl(method = "cv",
#                              number = 5,
#                              verboseIter = FALSE)


# set tuning
tune_grid <- expand.grid(
  .mtry = c(5, 7, 9),
  .splitrule = "variance",
  .min.node.size = c(5, 10)
)


# simpler model for model A (1)
set.seed(1234)
system.time({
  rf_model_1 <- train(
    formula_alt,
    data = data_work,
    method = "ranger",
    trControl = train_control,
    tuneGrid = tune_grid,
    importance = "impurity"
  )
})
rf_model_1
rf_model_1$bestTune
final_rf_rmse <- rf_model_1$results$RMSE[which.min(rf_model_1$results$RMSE)]
final_rf_rmse

summary(rf_model_1)

data_holdout_w_prediction <- data_holdout %>%
  mutate(predicted_price = predict(rf_model_1, newdata = data_holdout))
model2_rmse <- sqrt(sum((data_holdout_w_prediction$predicted_price-data_holdout_w_prediction$price_numeric)^2))
model2_rmse

#### Third model: Boosting #####
gbm_grid <-  expand.grid(interaction.depth = c(1, 5, 10), # complexity of the tree
                         n.trees = (4:10)*50, # number of iterations, i.e. trees
                         shrinkage = 0.1, # learning rate: how quickly the algorithm adapts
                         n.minobsinnode = 20 # the minimum number of training set samples in a node to commence splitting
)


set.seed(1234)
system.time({
  gbm_model <- train(formula_alt,
                     data = data_work,
                     method = "gbm",
                     trControl = train_control,
                     verbose = FALSE,
                     tuneGrid = gbm_grid)
})
gbm_model
gbm_model$bestTune

# Extracting the final RMSE
final_gbm_rmse <- gbm_model$results$RMSE[which.min(gbm_model$results$RMSE)]
final_gbm_rmse

data_holdout_w_prediction_new <- data_holdout %>%
  mutate(predicted_price = predict(gbm_model, newdata = data_holdout))
model3_rmse <- sqrt(sum((data_holdout_w_prediction_new$predicted_price-data_holdout_w_prediction_new$price_numeric)^2))
model3_rmse



insample_rmse <- c(lasso_cv_rmse[1, 1], final_rf_rmse, final_gbm_rmse)
holdout_rmse <- c(model1_rmse, model2_rmse, model3_rmse)
# all results in one table
main_table <- data.frame(rbind(insample_rmse, holdout_rmse))
colnames(main_table)<-c("Lasso", "Random Forest", "Boosting")
row.names(main_table) <- c("RMSE (insample)", "RMSE (holdout)")
main_table

stargazer(main_table, summary = F, digits=1, float=F, out="A1-results-table-Gushchin.tex")
stargazer(main_table, summary = F, digits=1, float=F, type="text",  out="A1-results-table-Gushchin.tex")

#=================== Task 2 =====================
# now we compare the model performance between two different dates
# first train the GBM model on full data listings_Vienna_09_2023.csv

set.seed(1234)
system.time({
  gbm_model2 <- train(formula_alt,
                     data = data,
                     method = "gbm",
                     trControl = train_control,
                     verbose = FALSE,
                     tuneGrid = gbm_grid)
})
gbm_model2
gbm_model2$bestTune

# Extracting the final RMSE
final_gbm2_rmse <- gbm_model2$results$RMSE[which.min(gbm_model2$results$RMSE)]
final_gbm2_rmse

# now process the data of listings_Vienna_12_2022.csv
data_original_old <- read.csv("listings_Vienna_12_2022.csv") # this is the latest data available
data_old <- data_original_old %>% select(-c(neighbourhood_group_cleansed, bathrooms, calendar_updated, license))
data_old %>% glimpse()

##### data processing & feature engineering ######
summary(data_old$accommodates) 
data_old <- data_old %>% filter(accommodates<7 & accommodates>1)
summary(data_old$accommodates) # now we have the data for places that can accommodate 2-6 people

data_old <- data_old %>% filter(room_type=="Entire home/apt") # because we are interested only in apartments
data_old %>% glimpse() 

table(data_old$property_type)
data_old <- data_old %>% filter(property_type=="Entire loft" | property_type=="Entire serviced apartment" | property_type=="Entire rental unit" | property_type=="Entire place") # these seem filter out non-apartments
data_old %>% glimpse() 

print(data_old$price[1]) # we need to clean this variable
data_old$price_cleaned <- gsub("[^0-9.]", "", data_old$price)
# Convert to numeric, handle NAs
data_old$price_numeric <- as.numeric(data_old$price_cleaned)
print(data_old$price_numeric[1]) # done!

summary(data_old$price_numeric)
# we need to treat the outliers 

#lets drop the outliers

sum(data_old$price_numeric > upper_limit)
sum(data_old$price_numeric < lower_limit)
(sum(data_old$price_numeric > upper_limit) + sum(data_old$price_numeric < lower_limit))/dim(data_old)[1]
#So there are 2.62% of observations that are less than 30 dollars per day or more than 400 dollars.
dim(data_old)
data_old <- data_old %>% filter(price_numeric<upper_limit & price_numeric>lower_limit)
dim(data_old)
# lets windsorize these observations and create dummies
data_old$outlier_high = 0
data_old$outlier_low = 0
#data_old$outlier_low[data_old$price_numeric < lower_limit] = 1
#data_old$outlier_high[data_old$price_numeric > upper_limit] = 1
#table(data_old$outlier_high)
#data_old$price_numeric[data_old$price_numeric < lower_limit] = lower_limit
#data_old$price_numeric[data_old$price_numeric > upper_limit] = upper_limit

# some columns we can drop because they don't carry any important information for our task
# listing_url, scrape_id, picture_url, host_url, host_picture_url, host_thumbnail_url
# Variables $name, $description, $neighborhood_overview, $host_about require text analysis
# will not use them because of the time constraint
#data <- data %>% select(-c(listing_url, scrape_id, picture_url, host_url, host_picture_url, host_thumbnail_url, price, id, last_scraped, name, description, neighborhood_overview, host_about, host_id, host_name, host_since, host_location, ))
data_old <- data_old %>% select(c(outlier_low, outlier_high, price_numeric, neighbourhood_cleansed,property_type,accommodates, bathrooms_text,bedrooms, beds, amenities)) 
data_old %>% glimpse() 

table(data_old$neighbourhood_cleansed) # this one is much cleaner than neighbourhood, better to use it
#clean the titles of the districts
data_old$neighbourhood_cleansed[data_old$neighbourhood_cleansed == "Rudolfsheim-F\u009fnfhaus"] = "Rudolfsheim-Fünfhaus"
data_old$neighbourhood_cleansed[data_old$neighbourhood_cleansed == "W\u008ahring"] = "Währing"
data_old$neighbourhood_cleansed[data_old$neighbourhood_cleansed == "Landstra§e"] = "Landstraße"
data_old$neighbourhood_cleansed[data_old$neighbourhood_cleansed == "D\u009abling"] = "Döbling"

table(data_old$bathrooms_text)
data_old$bathrooms_text[data_old$bathrooms_text == ""] = "0 baths"
# bathrooms_text also important (can try making it continuous)
data_old <- data_old %>% mutate(bathroom = ifelse(bathrooms_text == "0 baths", 0, 
                                          ifelse(bathrooms_text == "Half-bath", 0.5, 
                                                 ifelse(bathrooms_text == "1 bath", 1, 
                                                        ifelse(bathrooms_text == "1.5 baths", 1.5, 
                                                               ifelse(bathrooms_text == "2 baths", 2,
                                                                      ifelse(bathrooms_text == "2.5 baths", 2.5,
                                                                             ifelse(bathrooms_text == "3 baths", 3,
                                                                                    ifelse(bathrooms_text == "3.5 baths", 3.5,
                                                                                           ifelse(bathrooms_text == "4 baths", 4,
                                                                                                  ifelse(bathrooms_text == "4.5 baths", 4.5,5 )))))))))))
table(data_old$bathroom) #will use this variable

table(data_old$bedrooms) 
# lets regard NAs as zeros
data_old$bedrooms_flag <- 0
data_old$bedrooms_flag[is.na(data_old$bedrooms)] <- 1 #creating a flag variable
data_old$bedrooms[is.na(data_old$bedrooms)] <- 0

table(data_old$beds) 
# again lets regard NAs as zeros
data_old$beds_flag <- 0
data_old$beds_flag[is.na(data_old$beds)] <- 1 #creating a flag variable
data_old$beds[is.na(data_old$beds)] <- 0

# also the total size of amenity list can be an important variable
data_old$TV <- ifelse(grepl("TV", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$TV) 

data_old$wifi <- ifelse(grepl("Wifi", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$wifi) # very few don't have wifi

data_old$dishwasher <- ifelse(grepl("Dishwasher", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$dishwasher) 

data_old$coffeemaker <- ifelse(grepl("Coffee maker", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$coffeemaker) 

data_old$aircon <- ifelse(grepl("Air condition", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$aircon) 

data_old$microwave <- ifelse(grepl("Microwave", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$microwave) 

data_old$heating <- ifelse(grepl("heating", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$heating) 

data_old$fridge <- ifelse(grepl("Refrigerator", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$fridge) 

data_old$bathtub <- ifelse(grepl("Bathtub", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$bathtub) 

data_old$kitchen <- ifelse(grepl("kitchen", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$kitchen) 

data_old$pool <- ifelse(grepl("pool", data_old$amenities, ignore.case = TRUE), 1, 0)
table(data_old$pool) 

#help of Chat GPT#
# Create new variable amenity_length
data_old$amenity_length <- str_count(data_old$amenities, ",")
# Since the count gives the number of commas, add 1 to get the total number of amenities
data_old$amenity_length <- data_old$amenity_length + 1

summary(data_old$amenity_length) 

# List of column names you want to convert to factors
#columns_to_factor <- c("neighbourhood_cleansed", "property_type", "wifi", "TV", "dishwasher", "coffeemaker", "aircon", "microwave", "heating", "fridge", "bathtub", "kitchen", "pool", "bedrooms_flag", "beds_flag", "outlier_high", "outlier_low") #"bedrooms_outlier_flag", "beds_outlier_flag",

# Use lapply to apply as.factor to the specified columns
data_old[columns_to_factor] <- lapply(data_old[columns_to_factor], as.factor)
data_old %>% glimpse()
data_old <- data_old %>% mutate(accommodates_sq = accommodates^2, bathroom_sq = bathroom^2, beds_sq = beds^2, bedrooms_sq = bedrooms^2, amenity_length_sq = amenity_length^2) 
data_old <- data_old %>% mutate(accommodates_cub = accommodates^3, bathroom_cub = bathroom^3, beds_cub = beds^3, bedrooms_cub = bedrooms^3, amenity_length_cub = amenity_length^3) 
data_old <- data_old %>% mutate(accommodates_log = log(accommodates), amenity_length_log = log(amenity_length)) 

# now check the model's performance on listings_Vienna_12_2022.csv

data_holdout_w_prediction_old <- data_old %>%
  mutate(predicted_price = predict(gbm_model2, newdata = data_old))
model3_rmse <- sqrt(sum((data_holdout_w_prediction_old$predicted_price-data_holdout_w_prediction_old$price_numeric)^2))
model3_rmse
#=================== Task 3 =====================
##### Shapley values #####
#install.packages("devtools")
#devtools::install_github('ModelOriented/treeshap')
#install.packages("cli")
library(cli)
library(treeshap)

#define one-hot encoding function
dummy <- dummyVars(" ~ .", data=data_holdout$accommodates, fullRank=T, sep = NULL)

#perform one-hot encoding on data frame
data_holdout_ohe <- data.frame(predict(dummy, newdata=data_holdout$accommodates))

# replace "." character to " " to match model object names
names(data_holdout_ohe) <- gsub(x = names(data_holdout_ohe),
                                pattern = "\\.", 
                                replacement = " ")  

# unify model for treeshap
rf_model_unified <- ranger.unify(rf_model_1$finalModel, data_holdout_ohe)

treeshap_res <- treeshap(rf_model_unified, data_holdout_ohe[1:500, ])


## Download treeshap_fit.rds from OSF: https://osf.io/6p7r8
treeshap_res %>% write_rds("ch16-airbnb-random-forest/treeshap_fit.rds")

plot_contribution(treeshap_res, obs = 12)

plot_feature_importance(treeshap_res, max_vars = 10)


treeshap_inter <- treeshap(rf_model_unified, data_holdout_ohe[1:100, ], interactions = T)

#From Chat GPT
# Assuming data_holdout is your holdout dataset
observation_to_visualize <- data_holdout[26, , drop = FALSE]  # Replace 1 with the index of the observation you want to visualize

# Compute Shapley values
shap_values <- treeshap(rf_model_1, observation_to_visualize, type = "shapley")

unified_model <- ranger.unify(rf_model_1$finalModel, as.matrix(data_holdout))
treeshap1 <- treeshap(unified_model, head(data_holdout, 3))
plot_contribution(treeshap1, obs = 1)
treeshap1$shaps

