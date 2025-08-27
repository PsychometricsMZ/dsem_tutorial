# Library to manage file paths
library(here)

# Load the completed dataset from the specified path
dat <- readRDS(here::here("data", "completedata.RDS"))

ydata <- dat$data

num_obs <- dim(ydata)[1]
num_time <- dim(ydata)[2]
num_var <- dim(ydata)[3]

# Resort data so that BAI is in front
ydata_new <- array(NA, c(num_obs, num_time, num_var))
for (i in 1:num_obs){
  for (j in 1:num_time){
    ydata_new[i, j, ] <- ydata[i, j, c(10:12, 1:9)]
  }
}

# Standardize data based on first time point
ydata_std <- ydata_new
for (j in 1:num_time){
  for (k in 1:num_var){
    t1_mean <- mean(ydata_new[, 1, k], na.rm = TRUE)
    t1_std <- sd(ydata_new[, 1, k], na.rm = TRUE)
    ydata_std[, j, k] <- ((ydata_new[, j, k] - t1_mean) / t1_std)
  }
}

# Create a missingness indicator array (0 = observed, 1 = missing)
ymiss <- ydata_std
# Not missing -> 0
ymiss[!is.na(ydata_std)] <- 0
# Missing -> 1
ymiss[is.na(ydata_std)] <- 1

data <- list(ydata_std, ymiss)
names(data) <- c("data", "ymiss")

saveRDS(data, here::here("data", "finalcompletedata.RDS"))
