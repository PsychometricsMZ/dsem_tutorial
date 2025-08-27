# Library needed to run the jags code
library(R2jags)
# You might need to run install.packages('R2jags')
# and install.packages('rjags')

# Library to manage file paths
library(here)
# You might need to run install.packages("here")

# Import functions to standardize the factor loadings
source(here::here("utils/standardization.R"))
# You might need to run install.packages("expm")

################################################################################
# CONTENT of this illustration and additional files needed:
#
# - DATA preparation
# -> finalcompletedata.RDS
#
# - MODEL: One-factor state-switching AR(1) with person-specific random
#          intercept
# -> dlcsem_model1_1factor_ar1_2level-intslop.txt
#
# - PLOTS
#
# - CONVERGENCE checks
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

# Prepare data for one factor
data_1factor <- list(
  N = num_obs, # sample size
  Nt = num_time, # time points
  y = y_data[, , 1:3], # responses to items 1 to 3
  psi0 = diag(2), # hyperprior wishart distribution
  mu.zeta2 = rep(0, 2)  # hyperprior multivariate normal distribution
)

# Define the factor structure for standardization
factor_structure <- data.frame(
  indicator = 1:3,
  factor = rep(1, 3),
  loading_id = c(NA, 1, 2)  # first is fixed
)

################################################################################
# MODEL:  One-factor state-switching AR(1) with person-specific random intercept
################################################################################

# Define parameters which are being estimated
params_model1 <- c(
  "alpha.S1", "alpha.S2", "delta.alpha", "beta.S1", "beta.S2",
  "lambda.y", "nu.y", "sigma.y", "sigma.zeta21",
  "sigma.eta.S1", "sigma.eta.S2", "P2", "b2"
)
# alpha.S1: first AR(1) parameter of state 1 (random intercept/baseline value)
# alpha.S2: first AR(1) parameter of state 2 (random intercept/baseline value)
# delta.alpha: difference between the two AR(1) parameters
# beta.S1: second AR(1) parameter in state 1 (slope/autoregressive coefficient)
# beta.S2: second AR(1) parameter in state 2 (slope/autoregressive coefficient)
# lambda.y: factor loadings
# nu.y: factor intercepts
# sigma.y: variance of the residuals
# sigma.zeta21: variance of the random intercept
# sigma.eta.S1: variance of the latent state 1
# sigma.eta.S2: variance of the latent state 2
# P2: transition probabilities from state 2
# b2: parameter for the transition probabilities from state 2

# Run jags model for items 10-12 (all BAI indicators)
model1 <- jags.parallel(
  data = data_1factor, # data to use
  parameters.to.save = params_model1, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("05_dlcsem/models",
                          "dlcsem_model1_1factor_ar1_2level-intslop.txt")
)

# Store parameter estimates
est1 <- model1$BUGSoutput$summary
# Print parameter estimates
round(est1, 2)
#-------------------------------------------------------------------------------
# Inference for Bugs model at "dlcsem_model1_1factor_ar1_2level-intslop.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 20000 iterations saved. Running time = secs
#                  mean    sd    2.5%     25%     50%     75%   97.5% Rhat n.eff
# P2[1]            0.02  0.01    0.00    0.01    0.02    0.03    0.04 1.06    65
# P2[2]            0.98  0.01    0.96    0.97    0.98    0.99    1.00 1.04    80
# alpha.S1        -0.04  0.08   -0.19   -0.10   -0.04    0.02    0.12 1.17    20
# alpha.S2        -0.60  0.08   -0.74   -0.66   -0.61   -0.55   -0.43 1.10    31
# delta.alpha      0.56  0.06    0.45    0.53    0.57    0.60    0.68 1.02   180
# beta.S1          0.38  0.17    0.05    0.25    0.38    0.50    0.70 1.02   170
# beta.S2          0.53  0.10    0.33    0.46    0.53    0.60    0.74 1.05    64
# b2[1]            0.90  0.31    0.27    0.69    0.91    1.12    1.49 1.13    26
# b2[2]            0.46  0.34    0.02    0.19    0.39    0.66    1.29 1.05    65
# lambda.y[1]      0.86  0.05    0.76    0.83    0.86    0.89    0.96 1.01   210
# lambda.y[2]      1.35  0.07    1.22    1.30    1.35    1.40    1.49 1.06    48
# nu.y[1]          0.03  0.04   -0.04    0.01    0.03    0.05    0.10 1.01   600
# nu.y[2]          0.10  0.05    0.01    0.07    0.10    0.13    0.19 1.03    90
# sigma.eta.S1     0.05  0.01    0.04    0.04    0.05    0.05    0.06 1.01   350
# sigma.eta.S2     0.05  0.01    0.04    0.04    0.05    0.05    0.06 1.01   350
# sigma.y[1]       0.28  0.02    0.24    0.26    0.28    0.29    0.32 1.03    99
# sigma.y[2]       0.34  0.03    0.29    0.32    0.34    0.36    0.40 1.04    63
# sigma.y[3]       0.23  0.04    0.16    0.20    0.23    0.26    0.31 1.10    31
# sigma.zeta21[1,1]0.29  0.07    0.18    0.24    0.29    0.34    0.46 1.02   170
# sigma.zeta21[2,1]0.10  0.04    0.03    0.07    0.10    0.12    0.19 1.02   120
# sigma.zeta21[1,2]0.10  0.04    0.03    0.07    0.10    0.12    0.19 1.02   120
# sigma.zeta21[2,2]0.13  0.04    0.07    0.10    0.12    0.15    0.23 1.02   140
#
# SOME INTERPRETATION
# - alpha.S1: No change in anxiety
# - alpha.S2: Strong anxiety decline
# - beta: High temporal stability across both states
# - P2[1]: Low chance of reverting from S2 to S1
# - P2[2]: High persistence in S2
# - b2[2]: Lower anxiety levels predict transition to S2
#-------------------------------------------------------------------------------

# Calculate standardized factor loadings for state 1
# (see code in utils/standardization.R)
result_state1 <- run_factor_analysis_map(
  est = est1,
  factor_structure = factor_structure,
  n_factors = 1,
  beta_prefix = "beta.S1",
  sigma_eta_prefix = "sigma.eta.S1",
  sigma_y_prefix = "sigma.y",
  lambda_prefix = "lambda.y",
  person_intercept_var_name = "sigma.zeta21[1,1]",
  person_slope_var_name = "sigma.zeta21[2,2]"
)
print_analysis_results(result_state1)
#-------------------------------------------------------------------------------
# Standardized Factor Loadings:
#   indicator_1 indicator_2 indicator_3
# 1       0.795       0.713       0.889
#
# Intraclass Correlation Coefficient (ICC):
# 0.627
#
# Total Latent Variance (Factor Level):
#   0.478
#   Splitted into:
#     Raw Dynamic (ARMA) Factor Variance:
#       0.054
#     Raw Person-Level Intercept Variance:
#       0.295
#     Raw Person-Level Slope Variance:
#       0.129
#
# SOME INTERPRETATION
# -> factor loadings look good
#-------------------------------------------------------------------------------

# Calculate standardized factor loadings for state 2
result_state2 <- run_factor_analysis_map(
  est = est1,
  factor_structure = factor_structure,
  n_factors = 1,
  beta_prefix = "beta.S2",
  sigma_eta_prefix = "sigma.eta.S2",
  sigma_y_prefix = "sigma.y",
  lambda_prefix = "lambda.y",
  person_intercept_var_name = "sigma.zeta21[1,1]",
  person_slope_var_name = "sigma.zeta21[2,2]"
)
print_analysis_results(result_state2)
#-------------------------------------------------------------------------------
# Standardized Factor Loadings:
#   indicator_1 indicator_2 indicator_3
# 1       0.798       0.717       0.891
#
# Intraclass Correlation Coefficient (ICC):
# 0.632
#
# Total Latent Variance (Factor Level):
#   0.489
#   Splitted into:
#     Raw Dynamic (ARMA) Factor Variance:
#       0.065
#     Raw Person-Level Intercept Variance:
#       0.295
#     Raw Person-Level Slope Variance:
#       0.129
# SOME INTERPRETATION
# -> factor loadings look good
#-------------------------------------------------------------------------------


################################################################################
# PLOTS
################################################################################

# Define parameters to estimate for plotting
params_model1_plots <- c(
  "eta.S1", "eta.S2", "PS", "S"
)
# eta.S1: factor scores for state 1
# eta.S2: factor scores for state 2
# PS: transition probabilities
# S: state membership (1 or 2)

# Re-run jags model for items 10-12 (all BAI indicators)
model1_plots <- jags.parallel(
  data = data_1factor, # data to use
  parameters.to.save = params_model1_plots, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("05_dlcsem/models",
                          "dlcsem_model1_1factor_ar1_2level-intslop.txt")
)

fscores_illu5 <- model1_plots$BUGSoutput$summary
# save summary results for factor scores
#saveRDS(fscores_illu5,"05_dlcsem/results/results_illustration05_factorscores.RDS")

# if not run, read in the results
fscores_illu5 <- readRDS("05_dlcsem/results/results_illustration05_factorscores.RDS")


# Generate variable names for parameters at time t = 1
names_s       <- paste0("S[", 1:num_obs, ",1]")
names_ps      <- c()
names_eta_s1  <- paste0("eta.S1[", 1:num_obs, ",1]")
names_eta_s2  <- paste0("eta.S2[", 1:num_obs, ",1]")

# Append names for time t = 2 to num_time
for (j in 2:num_time) {
  names_s      <- c(names_s, paste0("S[", 1:num_obs, ",", j, "]"))
  names_ps     <- c(names_ps, paste0("PS[", 1:num_obs, ",", j, ",1]"))
  names_eta_s1 <- c(names_eta_s1, paste0("eta.S1[", 1:num_obs, ",", j, "]"))
  names_eta_s2 <- c(names_eta_s2, paste0("eta.S2[", 1:num_obs, ",", j, "]"))
}

# Extract posterior summaries from model output
s_hat <- matrix(model1_plots$BUGSoutput$summary[names_s, "50%"],
                nrow = num_obs, ncol = num_time, byrow = FALSE)
ps_hat <- matrix(model1_plots$BUGSoutput$summary[names_ps, "mean"],
                 nrow = num_obs, ncol = num_time - 1, byrow = FALSE)
ps_hat <- cbind(1, ps_hat)
eta_s1 <- matrix(model1_plots$BUGSoutput$summary[names_eta_s1, "mean"],
                 nrow = num_obs, ncol = num_time, byrow = TRUE)
eta_s2 <- matrix(model1_plots$BUGSoutput$summary[names_eta_s2, "mean"],
                 nrow = num_obs, ncol = num_time, byrow = TRUE)

########################## State-swiching over time ############################

pdf(
  file = here::here("05_dlcsem/plots",
                    "stateswitch_DLCSEM.pdf"),
  width = 12, height = 6
)

# Set layout for side-by-side plots
par(mfrow = c(1, 2),
    mar = c(5, 4.5, 4, 1),
    oma = c(0, 0, 0, 0),
    xpd = NA)

# LEFT PLOT: 1 - PShat (estimated P(S_t = 2)) over time
par(new = FALSE)
plot(1 - ps_hat[1, ], type = "l", ylim = c(0, 1), axes = FALSE,
     ylab = expression(hat(P)(S == 2)), xlab = "Session",
     col = "lightgray", lty = 3)

par(new = TRUE)
for (i in 2:N) {
  plot(1 - ps_hat[i, ], type = "l", ylim = c(0, 1), axes = FALSE,
       ylab = "", xlab = "", col = "lightgray", lty = 3)
  par(new = TRUE)
}

# Plot the average across individuals
plot(apply(1 - ps_hat, 2, mean), type = "l", ylim = c(0, 1), axes = FALSE,
     ylab = "", xlab = "")
axis(1)
axis(2, c(0, 1))

# RIGHT PLOT: Jittered state estimates (S_hat)
par(new = FALSE)
plot(jitter(s_hat[1, ], factor = 0.2), type = "l", ylim = c(0.9, 2.1),
     axes = FALSE, ylab = expression(hat(S)), xlab = "Session",
     col = "lightgray", lty = 3)

par(new = TRUE)
for (i in 2:N) {
  plot(jitter(s_hat[i, ], factor = 0.2), type = "l", ylim = c(0.9, 2.1),
       axes = FALSE, ylab = "", xlab = "", col = "lightgray", lty = 3)
  par(new = TRUE)
}

# Plot the average state estimate across individuals
plot(apply(s_hat == 2, 2, mean) + 1, type = "l", ylim = c(0.9, 2.1),
     axes = FALSE, ylab = "", xlab = "Session")

axis(1)
axis(2, c(1, 2), labels = c("1", "2"))

dev.off()


############# Individual trajectories centered at switching point ##############

# Two versions:
# 1. Uses hard state assignments (Shat) to weight eta.S1 / eta.S2
# 2. Uses estimated probabilities (PShat) to compute weighted average

# Note: Some individuals switch back temporarily (e.g., i = 1)
# -> compute each individual's first switch point (S transitions from 1 to 2)
switch_points <- rep(NA, num_obs)
for (i in 1:num_obs) {
  for (j in 2:num_time) {
    if (s_hat[i, j] > s_hat[i, j - 1]) {  # Detect a switch from state 1 to 2
      switch_points[i] <- j
      break  # Stop at the first switch point
    }
  }
}
# Prepare data matrices centered around each individual's switch point
x_vals_save <- y_vals_save <- matrix(NA, num_obs, num_time)

for (i in 1:num_obs) {
  if (!is.na(switch_points[i])) {
    x_vals_save[i, ] <- 1:num_time - switch_points[i] - 0.5
    y_vals_save[i, ] <- (eta_s1[i, ] * (s_hat[i, ] == 1) +
                           eta_s2[i, ] * (s_hat[i, ] == 2))
  }
}
# Define a common x-axis and compute the average trajectory across individuals
x_seq <- (-15:16) - 0.5
ymean <- c()
for (j in 1:length(x_seq)) {
  ymean[j] <- mean(y_vals_save[x_vals_save == x_seq[j]], na.rm = TRUE)
}

# === Save standalone DLC-SEM with states plot ===
pdf(
  file = here::here("05_dlcsem/plots",
                    "individual_trajectories_DLCSEM_states.pdf"),
  width = 10, height = 10
)

# Set plotting parameters
par(mfrow = c(1, 1))
par(new = FALSE)

for (i in 1:num_obs) {
  if (!is.na(switch_points[i])) {
    x_vals <- 1:num_time - switch_points[i] - 0.5
    y_vals <- (eta_s1[i, ] * (s_hat[i, ] == 1) +
               eta_s2[i, ] * (s_hat[i, ] == 2))

    # Plot each individual's trajectory (no title)
    plot(x_vals, y_vals, type = "l", ylim = y_limits, xlim = x_limits,
         lty = 3, ylab = expression(eta),
         xlab = "Session centered at switching point",
         axes = FALSE)
    par(new = TRUE)
  }
}

# Add mean line
plot(x_seq, ymean, type = "l", ylim = y_limits, xlim = x_limits,
     lty = 1, lwd = 2, xlab = "", ylab = "", axes = FALSE)

abline(v = 0)
axis(1, at = seq(x_limits[1], x_limits[2], by = 2))
axis(2)

dev.off()

# === Save standalone DLC-SEM with probabilities plot  ===
pdf(
  file = here::here("05_dlcsem/plots",
                    "individual_trajectories_DLCSEM_probs.pdf"),
  width = 10, height = 10
)

# Set plotting parameters
par(mfrow = c(1, 1))
par(new = FALSE)

for (i in 1:num_obs) {
  if (!is.na(switch_points[i])) {
    x_vals <- 1:num_time - switch_points[i] - 0.5
    y_vals <- (eta_s1[i, ] * ps_hat[i, ] +
               eta_s2[i, ] * (1 - ps_hat[i, ]))

    # Plot each individual's trajectory (no title)
    plot(x_vals, y_vals, type = "l", ylim = y_limits, xlim = x_limits,
         lty = 3, ylab = expression(eta),
         xlab = "Session centered at switching point",
         axes = FALSE)
    par(new = TRUE)
  }
}

abline(v = 0)
axis(1, at = seq(x_limits[1], x_limits[2], by = 2))
axis(2)

# Close the file
dev.off()


################################################################################
# CONVERGENCE checks
################################################################################

############################ Convergence Model 1 ###############################

# Run jags model 1 with burn-in parameter set to 1
model1_conv <- jags.parallel(
  data = data_1factor, # data to use (t=1)
  parameters.to.save = params_model1, # parameters to output
  n.iter = 20000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file =  here::here("05_dlcsem/models",
                           "dlcsem_model1_1factor_ar1_2level-int.txt")
)

# Create Markov Chain Monte Carlo samples
samps1 <- as.mcmc(model1_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("05_dlcsem/convergence_checks",
               "dlcsem_model1_gelmanplot.pdf"))
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
pdf(file = here::here("05_dlcsem/convergence_checks",
                      "dlcsem_model1_xyplot.pdf"), pointsize = 6)
plot(samps1)
# When the model is converged after the burn-in phase,
# the trace should not exhibit any increasing or decreasing trends,
# the different chains (plotted in different colors) should mix well
# and the density plot should be smooth and have only a single peak.
dev.off() # reset layout

# -> The model seems to be converged after 10000 MCMC iterations.
