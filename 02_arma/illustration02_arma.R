# Library needed to run the jags code
library(R2jags)
# You might need to run install.packages('R2jags')
# and install.packages('rjags')

# Library to manage file paths
library(here)
# You might need to run install.packages("here")

################################################################################
# CONTENT of this illustration and additional files needed:
#
# - DATA preparation and illustration of time series
# -> finalcompletedata.RDS
#
# - MODEL 1: 1 Person, 1 Item with AR(1) structure
# -> time_series_model1_observed_ar1_1person.txt
#
#-  MODEL 2: 1 Person, 3 Items with 1 latent factor and AR(1) structure
# -> time_series_model2_1factor_ar1_1person.txt
#
# - CONVERGENCE checks
#
# - APPENDIX MODEL 1a: 1 Person, 1 Item with AR(2) structure
# -> time_series_model1_observed_ar2_1person.txt
#
# - APPENDIX MODEL 1b: 1 Person, 1 Item with ARMA(1,1) structure
# -> time_series_model1_observed_ar1ma1_1person.txt
#
# - APPENDIX MODEL 2a: 1 Person, 3 Items, 1 latent factor, ARMA(1,1) structure
# -> time_series_model1_1factor_ar1ma1_1person.txt
################################################################################


################################################################################
# DATA preparation
################################################################################

# Read in the data from the file
dat <- readRDS(here::here("data", "finalcompletedata.RDS"))
y_data <- dat$data

# Save the number participants (N=57)
num_obs <- dim(y_data)[1]
# Save the number of time points (num_time=15)
num_time <- dim(y_data)[2]
# The third dimension contains the answers to the 12 questionnaire
# items which are sorted as follows:
# - columns 1:3 items of BAI
# - columns 4:6 items of TASK (WAI sub-scale)
# - columns 7:9 items of GOAL (WAI sub-scale)
# - columns 10:12 items of BOND (WAI sub-scale)

# Select data of a single person (choosing person 8 as the data of this
# individual is complete a trend is clearly visible, see time series plot below)
y_data_p1 <- y_data[8, , ]

################ Illustration of time series for items 1 to 3 ##################

pdf(here::here("02_arma/plots", "bai_time_series.pdf"),
    width = 12, height = 5)
# Plot layout and style
par(family = "Times", mfrow = c(1, 3), oma = c(2, 1, 2, 1),
    cex.axis = 1.5, cex.lab = 2, cex.main = 2.2)

for (j in 1:3) {
  par(mar = c(4, ifelse(j == 1, 5, 1), 3, 1))
  plot(NA, ylim = c(-2, 2), xlim = c(1, num_time),
       xlab = "measurement occasion t",
       ylab = ifelse(j == 1, "outcome y", ""),
       main = paste0("BAI ", j), axes = FALSE)
  axis(1)
  axis(2)
  lines(y_data_p1[, j], col = 1, lwd = 4)
}

# Add legend
par(mfrow = c(1, 1), new = TRUE)
plot(0, 0, type = "n", axes = FALSE, xlab = "",
     ylab = "", xlim = c(0, 1), ylim = c(0, 1))
legend("bottom", inset = c(0, -1.2),
       legend = c("Person 1", "Person 2", "Person 3",
                  "...", paste0("Person ", num_obs)),
       col = c(1, 2, 3, "white", num_obs),
       lty = 1, lwd = c(2, 1, 1, 1, 1),
       horiz = TRUE, bty = "n", xpd = NA, cex = 0.7)
dev.off()

####################### Prepare observed data (1 item) #########################

# Prepare data with one item
data_observed <- list(
  Nt = num_time, # number of time points
  y = y_data_p1[, 1] # item responses BAI indicator 1
)

####################### Prepare latent data (3 items) ##########################

# Prepare data with three items
data_latent <- list(
  Nt = Nt, # number of time points
  y = y_data_p1[, 1:3] # outcome BAI indicators 1 to 3
)

################################################################################
# MODEL 1: 1 Person, 1 Item with AR(1) structure
################################################################################

# Define parameters which are being estimated:
params_observed <- c("alpha", "beta", "sigma.y")
# alpha: first AR(1) parameter (intercept/baseline value)
# beta: second AR(1) parameter (slope/autoregressive coefficient)
# sigma.y: variance of the residuals

# Run jags model for item 1 (1st BAI indicator)
# with pre-selected burn-in (see convergence check below, APPENDIX 2.2)
model1 <- jags.parallel(
  data = data_observed, # data to use
  parameters.to.save = params_observed, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("02_arma/models",
                          "time_series_model1_observed_ar1_1person.txt")
)

# Store parameter estimates
est1 <- model1$BUGSoutput$summary
# Print parameter estimates
round(est1, 2)
################################################################################
# Inference for Bugs model "time_series_model1_observed_ar1_1person.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
# The output contains the mean of the parameters, their standard deviation (sd),
# several quantiles of their posterior distribution in %,
# the Gelman-Rubin statistic / potential scale reduction factor Rhat
# and a crude measure of effective sample size n.eff.
#           mean   sd  2.5%   25%   50%   75% 97.5% Rhat n.eff
# alpha     0.03 0.60 -0.92 -0.38 -0.08  0.36  1.47    1  5600
# beta      0.65 0.25  0.10  0.49  0.69  0.85  0.99    1  9300
# sigma.y   0.53 0.21  0.25  0.38  0.48  0.62  1.06    1 20000
#
# SOME INTERPRETATION
# -> the intercept alpha is close to zero
# -> the autocorrelation coefficient beta is high (0.65) and significant
#    according to the 95% credible interval
# -> the residual variance sigma.y is moderate (0.53)
################################################################################

# Check for the minimal effective sample size (validate convergence)
min(est1[est1[, "n.eff"] > 1, "n.eff"])
# output: 5600 -> high enough for stable estimates -> good convergence

# Check for smallest Rhat statistic (validate convergence)
max(est1[est1[, "n.eff"] > 1, "Rhat"])
# output: 1.001346 -> below threshold of 1.1 -> good convergence


################################################################################
# MODEL 2: 1 Person, 3 Items with 1 latent factor and AR(1) structure
################################################################################

# Define parameters which are being estimated:
params_latent <- c(
  "alpha", "beta", "lambda.y", "lambda.y.strd", "nu.y",
  "sigma.y", "sigma.eta"
)
# alpha: first AR(1) parameter (intercept/baseline value)
# beta: second AR(1) parameter (slope/autoregressive coefficient)
# lambda.y: factor loadings
# lambda.y.strd: standardized factor loadings
# nu.y: factor intercepts
# sigma.y: variance of the residuals
# sigma.eta: variance of the latent factor

# Run jags model for items 1-3 (all BAI indicators)
# with pre-selected burn-in (see convergence check below, APPENDIX 2.2)
model2 <- jags.parallel(
  data = data_latent, # data to use
  parameters.to.save = params_latent, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("02_arma/models",
                          "time_series_model2_1factor_ar1_1person.txt")
)
# file containing the model
# Store parameter estimates
est2 <- model2$BUGSoutput$summary
# Print parameter estimates
round(est2, 2)
################################################################################
# Inference for Bugs model "time_series_model2_1factor_ar1_1person.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
# The output contains the mean of the parameters, their standard deviation (sd),
# several quantiles of their posterior distribution in %,
# the Gelman-Rubin statistic / potential scale reduction factor Rhat
# and a crude measure of effective sample size n.eff.
#                     mean   sd  2.5%   25%   50%   75% 97.5% Rhat n.eff
# alpha               0.23 0.67 -0.88 -0.26  0.15  0.65  1.70    1  3100
# beta                0.78 0.21  0.23  0.68  0.85  0.94  0.99    1  3700
# lambda.y[1]         1.06 0.36  0.39  0.83  1.04  1.29  1.83    1 20000
# lambda.y[2]         1.04 0.28  0.54  0.85  1.02  1.21  1.64    1 20000
# lambda.y.strd[1]    0.74 0.10  0.53  0.68  0.75  0.81  0.90    1 20000
# lambda.y.strd[2]    0.62 0.16  0.24  0.52  0.64  0.74  0.87    1 20000
# lambda.y.strd[3]    0.74 0.13  0.44  0.67  0.77  0.83  0.92    1 20000
# nu.y[1]             1.19 0.29  0.65  1.01  1.19  1.37  1.78    1  3000
# nu.y[2]             0.59 0.22  0.16  0.44  0.58  0.73  1.05    1  2600
# sigma.eta           0.40 0.20  0.16  0.26  0.35  0.48  0.90    1  6800
# sigma.y[1]          0.30 0.14  0.13  0.20  0.27  0.36  0.67    1 14000
# sigma.y[2]          0.62 0.29  0.25  0.42  0.56  0.74  1.33    1 20000
# sigma.y[3]          0.29 0.14  0.12  0.20  0.26  0.35  0.65    1 20000
#
# SOME INTERPRETATION
# -> intercept alpha is larger than when using only one item,
#    but credible interval still covers zero
# -> the autocorrelation coefficient beta even higher (0.78) and credible
#    interval even narrower
# -> factor loadings all moderate to high
################################################################################

# Check for the minimal effective sample size (validate convergence)
min(est2[est2[, "n.eff"] > 1, "n.eff"])
# output: 2600 -> high enough for stable estimates -> good convergence

# Check for smallest Rhat statistic (validate convergence)
max(est2[est2[, "n.eff"] > 1, "Rhat"])
# output: 1.001877 -> below threshold of 1.1 -> good convergence


################################################################################
# CONVERGENCE Checks
################################################################################

############################ Convergence model 1 ###############################

# Run jags model for observed outcome with burn-in parameter set to 1
model1_conv <- jags.parallel(
  data = data_observed, # data to use
  parameters.to.save = params_observed, # parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file = here::here("02_arma/models",
                          "time_series_model1_observed_ar1_1person.txt")
)
# file containing the model

# Create Markov Chain Monte Carlo samples
samps1 <- as.mcmc(model1_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("02_arma/convergence_checks",
               "arma_model1_gelmanplot.pdf"))
par(new = FALSE)
# Loop through each parameter in parameters.to.save
for (v in 1:nvar(samps1)) {
  # Plot potential scale reduction factors against MCMC iterations
  gelman.plot(samps1[, v], ylim = c(0, 7), col = c("black", NA))
  # Ignore warning as they occur in the first iterations which we
  # will remove as burn-in phase.
  par(new = TRUE)
}
# Draw horizontal reference line at 1.1
# When the model is converged after the burn-in phase,
# the shrink factors should be below this threshold.
abline(h = 1.1)
dev.off() # reset layout

# Create trace plots and density plots for each parameter
pdf(file = here::here("02_arma/convergence_checks",
                      "arma_model1_xyplot.pdf"), pointsize = 6)
plot(samps1)
# When the model is converged after the burn-in phase,
# the trace should not exhibit any increasing or decreasing trends,
# the different chains (plotted in different colors) should mix well
# and the density plot should be smooth and have only a single peak.
dev.off() # reset layout

# -> The model seems to be converged after 5000 MCMC iterations.

############################ Convergence model 2 ###############################

# Run jags model for latent outcome with burn-in parameter set to 1
model2_conv <- jags.parallel(
  data = data_latent, # data to use (t=10)
  parameters.to.save = params_latent, # parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file = here::here("02_arma/models",
                          "time_series_model2_1factor_ar1_1person.txt")
)
# file containing the model

# Create Markov Chain Monte Carlo samples
samps2 <- as.mcmc(model2_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("02_arma/convergence_checks",
               "arma_model2_gelmanplot.pdf"))
par(new = FALSE)
for (v in 1:nvar(samps2)) {
  # Plot potential scale reduction factors against MCMC iterations
  gelman.plot(samps2[, v], ylim = c(0, 7), col = c("black", NA))
  par(new = TRUE)
}
# Draw horizontal reference line at 1.1
abline(h = 1.1)
dev.off() # reset layout

# Create trace plots and density plots for each parameter
pdf(file = here::here("02_arma/convergence_checks",
                      "arma_model2_xyplot.pdf"), pointsize = 6)
plot(samps2)
dev.off() # reset layout

# -> The model seems to be converged after 5000 MCMC iterations.



################################################################################
# APPENDIX MODEL 1a: AR(2) for observed outcome
################################################################################

# Define parameters which are being estimated:
params_observed_ar <- c("alpha", "beta", "sigma.y")
# alpha: first AR(1) parameter (intercept/baseline value)
# beta: second AR(1) parameter (slope/autoregressive coefficient)
# sigma.y: variance of the residuals

# Run jags model for item 10 (1st BAI indicator)
# with pre-selected burn-in (see convergence check below, APPENDIX 2.2)
model1a <- jags.parallel(
  data = data_observed, # data to use
  parameters.to.save = params_observed_ar, # parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("02_arma/models/appendix",
                          "time_series_model1a_observed_ar2_1person.txt")
)
# file containing the model
# Store parameter estimates
est1a <- model1a$BUGSoutput$summary
# Print parameter estimates
round(est1a, 2)
################################################################################
# Inference for Bugs model "time_series_model1a_observed_ar2_1person.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
#           mean   sd  2.5%   25%   50%   75% 97.5% Rhat n.eff
# alpha     0.13 0.66 -0.86 -0.35 -0.02  0.54  1.62    1 20000
# beta[1]   0.61 0.25  0.04  0.46  0.64  0.81  0.98    1  6800
# beta[2]   0.06 0.35 -0.63 -0.18  0.06  0.30  0.74    1 20000
# sigma.y   0.55 0.24  0.26  0.39  0.50  0.65  1.17    1 20000
#
# SOME INTERPRETATION
# - autoregressive effects now modeled by two beta parameters, their total
#   effect size is about the same as the one of a single beta in the AR(1) model
# - beta[2] is small with a large credibel interval
# - credible intervals of the other parameters bigger than in the AR(1) model
# - unexplained residual variance is larger, even though the model 
#   has an additional degree of freedom
# -> additional lag does not seem to be needed (overparametrization)
################################################################################


# Some additional diagnostics
loglik_matrix <- as.matrix(est1a, vars = "loglik0")

# Calculate WAIC
# Step 1: Compute pointwise log-likelihood mean and variance across MCMC samples
loglik_mean <- rowMeans(loglik_matrix)
loglik_var <- apply(loglik_matrix, 1, var)

# Step 2: Compute WAIC components
lppd <- sum(log(loglik_mean)) # Log pointwise predictive density
p_waic <- sum(loglik_var) # Effective number of parameters

# Step 3: Calculate WAIC
waic <- -2 * (lppd - p_waic)
waic

library(loo)
# You may need to run install.packages("loo")

# Convert the log-likelihood matrix into an appropriate format for loo
loo_result <- loo(loglik_matrix)

# Extract the LOO information criterion
loo_ic <- loo_result$looic
loo_ic


################################################################################
# APPENDIX MODEL 1b: ARMA(1,1) for observed outcome
################################################################################

# Define parameters which are being estimated:
params_observed_arma <- c("alpha", "beta", "gamma", "sigma.y")
# alpha: first AR(1) parameter (intercept/baseline value)
# beta: second AR(1) parameter (autoregressive coefficients)
# gamma: MA(1) parameter
# sigma.y: variance of the residuals

# Run jags model for item 10 (1st BAI indicator)
# with pre-selected burn-in (see convergence check below, APPENDIX 2.2)
model1b <- jags.parallel(
  data = data_observed, # data to use
  parameters.to.save = params_observed_arma, # parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file =  here::here("02_arma/models/appendix",
                           "time_series_model1b_observed_ar1ma1_1person.txt")
)
# file containing the model
# Store parameter estimates
est1b <- model1b$BUGSoutput$summary
# Print parameter estimates
round(est1b, 2)
################################################################################
# Inference for Bugs model "time_series_model1b_observed_ar1ma1_1person.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
#           mean   sd  2.5%   25%   50%   75% 97.5% Rhat n.eff
# alpha    -0.08 0.51 -0.89 -0.41 -0.17  0.16  1.20    1  5200
# beta      0.38 0.43 -0.52  0.07  0.43  0.74  0.98    1  1500
# gamma     0.31 0.42 -0.54  0.00  0.34  0.67  0.95    1  2400
# sigma.y   0.53 0.22  0.25  0.38  0.48  0.62  1.10    1 20000
#
# SOME INTERPRETATION
# - additional MA parameter gamma has considerable effect size
#   (however, with credible interval covering zero)
# - AR(1) parameters alpha and beta reduced compared to pure AR(1) model
# - unexplained residual variance slightly lower
# -> might be slightly better suited to model the time series
################################################################################


################################################################################
# APPENDIX MODEL 2a: ARMA(1,1) for latent variable
################################################################################

# Define parameters which are being estimated:
params_latent_arma <- c(
  "alpha", "beta", "gamma", "lambda.y", "lambda.y.strd",
  "nu.y", "sigma.y", "sigma.eta"
)
# alpha: first AR(1) parameter (intercept/baseline value)
# beta: second AR(1) parameter (autoregressive coefficients)
# gamma: MA(1) parameter
# lambda.y: factor loadings
# nu.y: factor intercepts
# sigma.y: variance of the residuals
# sigma.eta: variance of the latent factor

# Run jags model for items 10-12 (all BAI indicators)
# with pre-selected burn-in (see convergence check below, APPENDIX 2.2)
model2b <- jags.parallel(
  data = data_latent, # data to use
  parameters.to.save = params_latent_arma, # parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("02_arma/models/appendix",
                          "time_series_model2b_1factor_ar1ma1_1person.txt")
)
# file containing the model
# Store parameter estimates
est2b <- model2b$BUGSoutput$summary
# Print parameter estimates
round(est2b, 2)
################################################################################
# Inference for Bugs model "time_series_model1b_observed_ar1ma1_1person.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
#                     mean   sd  2.5%   25%   50%   75% 97.5% Rhat n.eff
# alpha               0.20 0.66 -0.86 -0.27  0.11  0.62  1.66    1 13000
# beta                0.67 0.32 -0.17  0.53  0.78  0.91  0.99    1 20000
# gamma               0.33 0.38 -0.46  0.07  0.36  0.63  0.95    1  9700
# lambda.y[1]         1.04 0.36  0.36  0.80  1.02  1.26  1.79    1  3000
# lambda.y[2]         1.03 0.27  0.54  0.85  1.01  1.19  1.62    1 20000
# lambda.y.strd[1]  0.74 0.10  0.53  0.68  0.75  0.81  0.90    1  5900
# lambda.y.strd[2]  0.60 0.17  0.22  0.50  0.62  0.73  0.87    1  5200
# lambda.y.strd[3]  0.74 0.13  0.43  0.67  0.76  0.83  0.92    1  9500
# nu.y[1]             1.18 0.29  0.64  0.99  1.17  1.36  1.78    1  8300
# nu.y[2]             0.58 0.23  0.15  0.44  0.58  0.73  1.05    1  3900
# sigma.eta           0.39 0.20  0.15  0.25  0.34  0.46  0.90    1  6600
# sigma.y[1]          0.30 0.14  0.13  0.20  0.27  0.35  0.64    1  5800
# sigma.y[2]          0.62 0.30  0.25  0.42  0.56  0.75  1.38    1  6500
# sigma.y[3]          0.29 0.14  0.12  0.20  0.26  0.35  0.64    1  5600
#
# SOME INTERPRETATION
# - factor structure the same as in AR(1) model
# - see interpretation of model 1b (ARMA(1,1) for observed outcome) for
#   the other parameters
################################################################################
