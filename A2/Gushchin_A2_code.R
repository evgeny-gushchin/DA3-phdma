### Assignment 2 Prediction with ML for Economists Fall 2023 ###
### by Evgeny Gushchin
rm(list = ls()) # cleaning the environment
#install.packages("installr")
#installr::updateR()
#install.packages("remotes")
#install.packages("skimr")
#install.packages("directlabels")
install.packages("cowplot")
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
data <- data %>% select(-c(listing_url, scrape_id, picture_url, host_url, host_picture_url, host_thumbnail_url, price))
data %>% glimpse() 

table(data$source) # not sure how this variable can be useful
g1 <- ggplot(data, aes(x = factor(source), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Source",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g1
# but the plots suggest that source matters
# basically the variable seems to show for how long the apartment was online (city scrape = new, previous scrape = old)

# Variables $name, $description, $neighborhood_overview, $host_about require text analysis
# will not use them because of the time constraint

table(data$host_location) # the host's location can be important for pricing
# we can create 
data$host_location_new <- ifelse(grepl("Vienna", data$host_location, ignore.case = TRUE), "Vienna",
                                 ifelse(grepl("Austria", data$host_location, ignore.case = TRUE), "Inside Austria",
                                        "Outside of Austria"))
table(data$host_location_new) 
g2 <- ggplot(data, aes(x = factor(host_location_new), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Host location",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g2
# judging by the graph the variable doesn't seem that important

table(data$host_response_time) # the missing value "" can be regarded as NA
data$host_response_time[data$host_response_time == ""] <- "N/A"
g3 <- ggplot(data, aes(x = factor(host_response_time), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Host location",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g3
# this variable seem to matter, could be a proxy of host's professionalism

table(data$host_response_rate) # N/A's can be trasfered into zeros as well as the blanck value
data$host_response_rate[data$host_response_rate %in% c("N/A","")] <- "0%"
# Help from ChatGPT
data$host_response_rate_numeric <- as.numeric(sub("%", "", data$host_response_rate))

summary(data$host_response_rate_numeric) 
g4 <- ggplot(data, aes(x=host_response_rate_numeric, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Host response rate",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g4
# the variable doesn't seem very informative (although 0% and 100% may be important)

table(data$host_acceptance_rate) # N/A's can be trasfered into zeros as well as the blanck value
data$host_acceptance_rate[data$host_acceptance_rate %in% c("N/A","")] <- "0%"
data$host_acceptance_rate_numeric <- as.numeric(sub("%", "", data$host_acceptance_rate))
summary(data$host_acceptance_rate_numeric) 
g5 <- ggplot(data, aes(x=host_acceptance_rate_numeric, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Host acceptance rate",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g5
# the variable doesn't seem very informative (although 0% and 100% may be important)
cor(data$host_response_rate_numeric, data$host_acceptance_rate_numeric) # very correlated, yet not identical

table(data$host_is_superhost) # the blanc values should be treated as false
data$host_is_superhost[data$host_is_superhost == ""] <- "f"
g6 <- ggplot(data, aes(x = factor(host_is_superhost), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Super Host?",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g6
# super hosts charge more

table(data$host_neighbourhood)
g7 <- ggplot(data, aes(x = factor(host_neighbourhood), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Host's neighbourhood",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g7
# host_neighbourhood may be relevant, but we need to find a way to reduce the size of this variable (make it continuous or regroup)

table(data$host_listings_count) #can be a proxy for individuals and rental companies
summary(data$host_listings_count)
table(data$host_total_listings_count)
cor(data$host_listings_count[!is.na(data$host_listings_count)], data$host_total_listings_count[!is.na(data$host_total_listings_count)])
# two variable are highly correlated between each other, we can use either of the two
# we can make this variable into factor and unite all observations with more than 9 listing into one group "10+"
# let the NA value become zero
data$host_listings_count[is.na(data$host_listings_count)] <- 0
data$host_listings_count[data$host_listings_count > 9] <- 10
data$host_listings_count[data$host_listings_count== 0] <- 1 # if host in our dataset then there should be at least one apartment online
table(data$host_listings_count) 
g8 <- ggplot(data, aes(x = factor(host_listings_count), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Host's listings count",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g8
# may be relevant

table(data$host_verifications)
g85 <- ggplot(data, aes(x = factor(host_verifications), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Host verification",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g85
# not sure this is very relevant for apartment pricing

table(data$host_has_profile_pic)
# let's assume the missing value is also "f"
data$host_has_profile_pic[data$host_has_profile_pic == ""] <- "f"
g9 <- ggplot(data, aes(x = factor(host_has_profile_pic), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Host has profile pic",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g9
# the variable may be somewhat relevant
# should not be relevant for a company

table(data$host_identity_verified)
# again let's assume the missing value is also "f"
data$host_identity_verified[data$host_identity_verified == ""] <- "f"
g10 <- ggplot(data, aes(x = factor(host_identity_verified), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Host has verified identity",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g10
# this variable is much more important for the pricing than the profile picture

table(data$neighbourhood) 
table(data$neighbourhood_cleansed) # this one is much cleaner better to use it
g11 <- ggplot(data, aes(x = factor(neighbourhood_cleansed), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Neighbourhood",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g11
# neighbourhood_cleansed is an important variable but we need to find ways to unite some groups or to make it continuous (mean income level?)

# I don't think I need latitude and longitude as I already have neighbourhoods and I don't do the mapping
data <- data %>% select(-c(latitude, longitude))
data %>% glimpse()

table(data$room_type) # this variable was used for filtering

table(data$property_type) # also was used for filtering but there are several groups left, that mey be helpful for prediction
g12 <- ggplot(data, aes(x = factor(property_type), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Property type",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g12
# property_type seems relevant

summary(data$accommodates)
g13 <- ggplot(data, aes(x = factor(accommodates), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "How many people can live",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g13
# accommodates is a very important variable (a proxy for apartment size)

table(data$bathrooms_text)
g14 <- ggplot(data, aes(x = factor(bathrooms_text), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "How many bathrooms",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g14
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
data$bedrooms[is.na(data$bedrooms)] <- 0
g15 <- ggplot(data, aes(x = factor(bedrooms), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Bedrooms",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g15
# bedrooms seems important (continuous)

table(data$beds) 
# again lets regard NAs as zeros
data$beds[is.na(data$beds)] <- 0
g16 <- ggplot(data, aes(x = factor(beds), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Beds",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g16

cor(data$bedrooms,data$beds) # correlation is 0.53
cor(data$accommodates,data$beds) # correlation is 0.68
cor(data$accommodates,data$bedrooms) # correlation is 0.56

print(data$amenities[1]) 
# the variable includes a list of amenities, need text analysis
# some amenities are seasonally important (aircon in summer and heating in winter)
# some are individually important (e.g. some people always prefer to have a dishwasher, or a coffee machine)
# let's claim that wifi is universally important amenity (lets create a dummy for it)
# also the total size of aminity list can be an imporant variable

data$wifi <- ifelse(grepl("Wifi", data$amenities, ignore.case = TRUE), 1, 0)
table(data$wifi) # very few don't have wifi
g17 <- ggplot(data, aes(x = factor(wifi), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Wifi",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g17
# wifi does seem important

#help of Chat GPT#
# Create new variable amenity_length
data$amenity_length <- str_count(data$amenities, ",")
# Since the count gives the number of commas, add 1 to get the total number of amenities
data$amenity_length <- data$amenity_length + 1

table(data$amenity_length) 
g18 <- ggplot(data, aes(x = factor(amenity_length), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Number of amenities",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g18
# amenity_length seems to be important (continuous) 

table(data$minimum_nights)
g19 <- ggplot(data, aes(x = factor(minimum_nights), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Minimum nights",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g19
# very nonlinear 

table(data$maximum_nights)
g20 <- ggplot(data, aes(x = factor(maximum_nights), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Maximum nights",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g20
# seems noisy, can try to use as continuous

#not sure how to use minimum_minimum_nights, maximum_minimum_nights, minimum_maximum_nights, maximum_maximum_nights, minimum_nights_avg_ntm, maximum_nights_avg_ntm 

#help of Chat GPT#
# Specify the variables for which you want to calculate correlations
selected_variables <- c(
  "minimum_nights",
  "minimum_minimum_nights",
  "maximum_minimum_nights",
  "minimum_nights_avg_ntm"
)
# Extract the selected variables from the data frame
selected_data <- data[, selected_variables]
# Calculate pairwise correlations
correlation_matrix1 <- cor(selected_data)
# Print the correlation matrix
print(correlation_matrix1)
# Specify the variables for which you want to calculate correlations
selected_variables <- c(
  "maximum_nights",
  "minimum_maximum_nights",
  "maximum_maximum_nights",
  "maximum_nights_avg_ntm"
)
# "maximum_maximum_nights" and "maximum_nights_avg_ntm" correlate to each other almost 100%
# Extract the selected variables from the data frame
selected_data <- data[, selected_variables]
# Calculate pairwise correlations
correlation_matrix2 <- cor(selected_data)
# Print the correlation matrix
print(correlation_matrix2)
# "minimum_maximum_nights", "maximum_maximum_nights" and "maximum_nights_avg_ntm" correlate to each other almost 100%

# let use then minimum_nights, minimum_minimum_nights, maximum_minimum_nights, maximum_nights and minimum_maximum_nights

table(data$has_availability)
g21 <- ggplot(data, aes(x = factor(has_availability), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Availability",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g21
# has_availability seems important

table(data$availability_30)
g22 <- ggplot(data, aes(x = factor(availability_30), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Availability 30",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g22
# availability_30 seems important

table(data$availability_60)
g23 <- ggplot(data, aes(x = factor(availability_60), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Availability 60",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g23

table(data$availability_90)
table(data$availability_365)
g24 <- ggplot(data, aes(x = factor(availability_365), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Availability 365",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g24
# availability_365 might be relevant

# Specify the variables for which you want to calculate correlations
selected_variables <- c(
  "availability_30",
  "availability_60",
  "availability_90",
  "availability_365"
)
# Extract the selected variables from the data frame
selected_data <- data[, selected_variables]
# Calculate pairwise correlations
correlation_matrix3 <- cor(selected_data)
# Print the correlation matrix
print(correlation_matrix3)

# it seems it makes sense to use both availability_30 and availability_365 as the least correlated

table(data$calendar_last_scraped)
g25 <- ggplot(data, aes(x = factor(calendar_last_scraped), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Calendar last scraped",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g25
# calendar_last_scraped should not be relevant

table(data$number_of_reviews)
g26 <- ggplot(data, aes(x=number_of_reviews, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Number of reviews",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g26
# number_of_reviews seems very noisy
selected_data <- data[, c("number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d")]
cor(selected_data)
# don't seem very correlated to each other

g27 <- ggplot(data, aes(x=number_of_reviews_ltm, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Number of reviews in the last year",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g27

g28 <- ggplot(data, aes(x=number_of_reviews_l30d, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Number of reviews in the last 30 days",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g28
g28a <- ggplot(data, aes(x = factor(number_of_reviews_l30d), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Number of reviews in the last 30 days",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g28a
# doesn't seem to matter very much

# Convert text representations to date variables
data$last_scraped <- as.Date(data$last_scraped)
data$first_review <- as.Date(data$first_review)
data$last_review <- as.Date(data$last_review)

data$days_since_first_review <- data$last_scraped - data$first_review
table(data$days_since_first_review)

g29 <- ggplot(data, aes(x=days_since_first_review, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Days since first review",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g29
#could be relevant

data$days_since_last_review <- data$last_scraped - data$last_review
table(data$days_since_last_review)
#cov(as.numeric(data$days_since_first_review), data$days_since_last_review)
g30 <- ggplot(data, aes(x=days_since_last_review, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Days since last review",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g30
# could be relevant
# but not sure how to treat missing values here

table(data$review_scores_rating)
g31 <- ggplot(data, aes(x=review_scores_rating, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Review rating",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g31
g31a <- ggplot(data, aes(x = factor(round(data$review_scores_rating,1)), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Review rating",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g31a

table(data$review_scores_value)
g32 <- ggplot(data, aes(x=review_scores_value, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Review value",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g32
g32a <- ggplot(data, aes(x = factor(round(data$review_scores_value,1)), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Review value",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g32a

# Specify the variables for which you want to calculate correlations
selected_variables <- c(
  "review_scores_rating",
  "review_scores_accuracy",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "review_scores_communication",
  "review_scores_location",
  "review_scores_value"
)
# Extract the selected variables from the data frame
selected_data <- data[, selected_variables]
# Chat GPT help
selected_data <- selected_data[complete.cases(selected_data), ]
# Calculate pairwise correlations
correlation_matrix4 <- cor(selected_data)
# Print the correlation matrix
print(correlation_matrix4) # somewhat correlated, but can use all

table(data$instant_bookable)
g33 <- ggplot(data, aes(x = factor(instant_bookable), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Bookable now?",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g33
# instant_bookable seems somewhat relevant

table(data$calculated_host_listings_count)
data$calculated_host_listings_count[data$calculated_host_listings_count > 9] <- 10
#may be relevant
g34 <- ggplot(data, aes(x = factor(calculated_host_listings_count), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Host's listings count",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g34
cor(data$calculated_host_listings_count, data$host_listings_count) # very correlated probably no need

table(data$calculated_host_listings_count_entire_homes)
#may be relevant
table(data$calculated_host_listings_count_private_rooms)
#doesn't seem very relevant
table(data$calculated_host_listings_count_shared_rooms)
#doesn't seem very relevant

summary(data$reviews_per_month) # it seems that the NA value here is simply 0
data$reviews_per_month_flag <- 0
data$reviews_per_month_flag[is.na(data$reviews_per_month)] <- 1 #creating a flag variable
data$reviews_per_month[is.na(data$reviews_per_month)] <- 0
summary(data$reviews_per_month)
g35 <- ggplot(data, aes(x = factor(reviews_per_month), y = price_numeric,
)) +
  geom_boxplot(alpha=0.8, na.rm=T, outlier.shape = NA, width = 0.8) +
  stat_boxplot(geom = "errorbar", width = 0.8, size = 0.3, na.rm=T)+
  labs(x = "Reviews per month",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))+
  theme_minimal()
g35
g35a <- ggplot(data, aes(x=reviews_per_month, y=price_numeric)) + 
  geom_point(alpha=0.05)+ theme_minimal() +
  labs(x = "Reviews per month",y = "Apartment price")+
  scale_y_continuous(expand = c(0.01,0.01), limits=c(0, 200), breaks = seq(0,200, 20))
g35a

# List of column names you want to convert to factors
columns_to_factor <- c("host_response_time", "host_is_superhost", "host_listings_count", "neighbourhood_cleansed", "property_type", "wifi")

# Use lapply to apply as.factor to the specified columns
data[columns_to_factor] <- lapply(data[columns_to_factor], as.factor)
data %>% glimpse()
data <- data %>% mutate(accommodates_sq = accommodates^2, bathroom_sq = bathroom^2, beds_sq = beds^2, bedrooms_sq = bedrooms^2, amenity_length_sq = amenity_length^2) #days_since_first_review_sq = days_since_first_review^2, days_since_last_review_sq = days_since_last_review^2)
continuous_vars <- c("accommodates","accommodates_sq","bathroom","bathroom_sq","beds","beds_sq","bedrooms","bedrooms_sq", "amenity_length", "amenity_length_sq") #"days_since_first_review", "days_since_first_review_sq","days_since_last_review","days_since_last_review_sq")

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
install.packages("devtools")
devtools::install_github('ModelOriented/treeshap')
install.packages("cli")
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

unified_model <- xgboost.unify(xgb_model, as.matrix(data))
treeshap1 <- treeshap(unified_model, head(data, 3))
plot_contribution(treeshap1, obs = 1)
treeshap1$shaps

