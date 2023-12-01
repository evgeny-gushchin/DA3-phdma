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

# some columns we can drop because they don't carry any important information for our task
# listing_url, scrape_id, picture_url, host_url, host_picture_url, host_thumbnail_url
# Variables $name, $description, $neighborhood_overview, $host_about require text analysis
# will not use them because of the time constraint
#data <- data %>% select(-c(listing_url, scrape_id, picture_url, host_url, host_picture_url, host_thumbnail_url, price, id, last_scraped, name, description, neighborhood_overview, host_about, host_id, host_name, host_since, host_location, ))
data <- data %>% select(c(price_numeric, neighbourhood_cleansed,property_type,accommodates, bathrooms_text,bedrooms, beds, amenities)) 
data %>% glimpse() 

table(data$neighbourhood_cleansed) # this one is much cleaner than neighbourhood, better to use it
g1 <- ggplot(data, aes(x = factor(neighbourhood_cleansed), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Neighbourhood",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g1
# neighbourhood_cleansed is an important variable but we need to find ways to unite some groups or to make it continuous (mean income level?)

# I don't think I need latitude and longitude as I already have neighbourhoods and I don't do the mapping
#data <- data %>% select(-c(latitude, longitude))
#data %>% glimpse()

#table(data$room_type) # this variable was used for filtering

table(data$property_type) # also was used for filtering but there are several groups left, that mey be helpful for prediction
g2 <- ggplot(data, aes(x = factor(property_type), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Property type",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g2
# property_type seems relevant

summary(data$accommodates)
g3 <- ggplot(data, aes(x = factor(accommodates), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "How many people can live",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g3
# accommodates is a very important variable (a proxy for apartment size)

table(data$bathrooms_text)
g4 <- ggplot(data, aes(x = factor(bathrooms_text), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "How many bathrooms",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g4
# bathrooms_text also important (can try making it continuous)
data <- data %>% mutate(bathroom = ifelse(bathrooms_text == "0 baths", 0, 
                                   ifelse(bathrooms_text == "Half-bath", 0.5, 
                                   ifelse(bathrooms_text == "1 bath", 1, 
                                   ifelse(bathrooms_text == "1.5 baths", 1.5, 
                                   ifelse(bathrooms_text == "2 baths", 2,
                                   ifelse(bathrooms_text == "2.5 baths", 2.5,
                                   ifelse(bathrooms_text == "3 baths", 3,3.5 ))))))))
table(data$bathroom) #will use this variable

table(data$bedrooms) 
# lets regard NAs as zeros
data$bedrooms_flag <- 0
data$bedrooms_flag[is.na(data$bedrooms)] <- 1 #creating a flag variable
data$bedrooms[is.na(data$bedrooms)] <- 0
g5 <- ggplot(data, aes(x = factor(bedrooms), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Bedrooms",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g5
# bedrooms seems important (continuous)

table(data$beds) 
# again lets regard NAs as zeros
data$beds_flag <- 0
data$beds_flag[is.na(data$beds)] <- 1 #creating a flag variable
data$beds[is.na(data$beds)] <- 0
g6 <- ggplot(data, aes(x = factor(beds), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Beds",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g6

print(data$amenities[1]) 
# the variable includes a list of amenities, need text analysis
# some amenities are seasonally important (aircon in summer and heating in winter)
# some are individually important (e.g. some people always prefer to have a dishwasher, or a coffee machine)

# also the total size of amenity list can be an important variable
data$TV <- ifelse(grepl("TV", data$amenities, ignore.case = TRUE), 1, 0)
table(data$TV) 
g7 <- ggplot(data, aes(x = factor(TV), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "TV",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g7

data$wifi <- ifelse(grepl("Wifi", data$amenities, ignore.case = TRUE), 1, 0)
table(data$wifi) # very few don't have wifi
g8 <- ggplot(data, aes(x = factor(wifi), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Wifi",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g8
# wifi does seem important

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

#help of Chat GPT#
# Create new variable amenity_length
data$amenity_length <- str_count(data$amenities, ",")
# Since the count gives the number of commas, add 1 to get the total number of amenities
data$amenity_length <- data$amenity_length + 1

summary(data$amenity_length) 
g9 <- ggplot(data, aes(x = factor(amenity_length), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Number of amenities",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g9
# amenity_length seems to be important (continuous) 

# List of column names you want to convert to factors
columns_to_factor <- c("neighbourhood_cleansed", "property_type", "wifi", "TV", "dishwasher", "coffeemaker", "aircon", "microwave", "heating", "fridge", "bathtub", "bedrooms_flag", "beds_flag")

# Use lapply to apply as.factor to the specified columns
data[columns_to_factor] <- lapply(data[columns_to_factor], as.factor)
data %>% glimpse()
data <- data %>% mutate(accommodates_sq = accommodates^2, bathroom_sq = bathroom^2, beds_sq = beds^2, bedrooms_sq = bedrooms^2, amenity_length_sq = amenity_length^2) 
data <- data %>% mutate(accommodates_cub = accommodates^3, bathroom_cub = bathroom^3, beds_cub = beds^3, bedrooms_cub = bedrooms^3, amenity_length_cub = amenity_length^3) 
data <- data %>% mutate(accommodates_log = log(accommodates), bathroom_log = log(bathroom), beds_log = log(beds), bedrooms_log = log(bedrooms), amenity_length_log = log(amenity_length)) 

continuous_vars <- c("accommodates","accommodates_sq","accommodates_cub","accommodates_log","bathroom","bathroom_sq","bathroom_cub", "bathroom_log","beds","beds_sq", "beds_cub", "beds_log","bedrooms","bedrooms_sq", "bedrooms_cub","bedrooms_log", "amenity_length", "amenity_length_sq", "amenity_length_cub", "amenity_length_log") 

#Look up property type interactions

# NB all graphs, we exclude  extreme values of price
datau <- subset(data, price_numeric<400)
# Create a scatter plot with color-coded points
ggplot(datau, aes(x = as.factor(accommodates), y = price_numeric, color = property_type)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5) +
  labs(x = "Accommodates",
       y = "Price Numeric") +
  theme_minimal()

ggplot(datau, aes(x = as.factor(accommodates), y = price_numeric, color = wifi)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5) +
  labs(x = "Accommodates",
       y = "Price Numeric") +
  theme_minimal()

ggplot(datau, aes(x = property_type, y = price_numeric, color = wifi)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5) +
  labs(x = "Property type",
       y = "Price Numeric") +
  theme_minimal()

# dummies suggested by graphs
X1  <- c("f_room_type*f_property_type",  "f_room_type*d_familykidfriendly")

# Additional interactions of factors and dummies
X2  <- c("d_airconditioning*f_property_type", "d_cats*f_property_type", "d_dogs*f_property_type")
X3  <- c(paste0("(f_property_type + f_room_type + f_cancellation_policy + f_bed_type) * (",
                paste(amenities, collapse=" + "),")"))

#################################
# Separate hold-out set #
#################################

# create a holdout set (20% of observations)
smp_size <- floor(0.2 * nrow(data))

# Set the random number generator: It will make results reproducable
set.seed(20180123)

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
# take model 8 (and find observations where there is no missing data)may
#vars_model_7 <- c("price_numeric", continuous_vars)
vars_model_1 <- c("price_numeric", continuous_vars, columns_to_factor)

# Set lasso tuning parameters
n_folds=5
train_control <- trainControl(method = "cv", number = n_folds)
tune_grid <- expand.grid("alpha" = c(1), "lambda" = seq(0.05, 1, by = 0.05))

# We use model 7 without the interactions so that it is easy to compare later to post lasso ols
formula <- formula(paste0("price_numeric ~ ", paste(setdiff(vars_model_1, "price_numeric"), collapse = " + ")))

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
predictions_holdout <- predict(lasso_model, data_holdout)
# look at holdout RMSE
model1_rmse <- sqrt(sum((predictions_holdout-data_holdout$price_numeric)^2))
model1_rmse
# strange it is too large (it seems we are overfitting)


#### Second model: Random Forest #####

# do 5-fold CV
train_control <- trainControl(method = "cv",
                              number = 5,
                              verboseIter = FALSE)


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
    formula,
    data = data_work,
    method = "ranger",
    trControl = train_control,
    tuneGrid = tune_grid,
    importance = "impurity"
  )
})
rf_model_1

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
  gbm_model <- train(formula,
                     data = data_work,
                     method = "gbm",
                     trControl = train_control,
                     verbose = FALSE,
                     tuneGrid = gbm_grid)
})
gbm_model

data_holdout_w_prediction_new <- data_holdout %>%
  mutate(predicted_price = predict(gbm_model, newdata = data_holdout))
model3_rmse <- sqrt(sum((data_holdout_w_prediction_new$predicted_price-data_holdout_w_prediction_new$price_numeric)^2))
model3_rmse

#=================== Task 2 =====================
# now we compare the model performance between two different dates

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

