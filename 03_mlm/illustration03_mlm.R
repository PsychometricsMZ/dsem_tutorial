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
# - DATA preparation
# -> finalcompletedata.RDS
#
# - MODEL 1: 2-level single item AR(1) with person-specific random
#            intercept and slope
# -> mlm1_observed_ar1_2level-intslop.txt
#
# - MODEL 2: cross-classified single item AR(1) with person-specific
#            intercept + slope and time-specific random slope
# -> mlm1_observed_ar1_cross.txt
#
# - CONVERGENCE checks
#
# - APPENDIX MODEL 1a: 2-level single item ARMA(1, 1) with person-specific
#                      random intercept and slope
# -> mlm1a_observed_ar1ma1_2level-intslop.txt
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

# Prepare data for jags
data_prepared <- list(
  N = num_obs, # sample size
  Nt = num_time, # time points
  y = y_data[, , 1], # repsonses to item 1
  psi0 = diag(2), # hyperprior wishart distribution
  mu.eps2 = rep(0, 2)  # hyperprior multivariate normal distribution
)

####################### Spaghetti plots for illustration #######################

pdf(here::here("03_mlm/plots", "bai_spaghetti_plots.pdf"),
    width = 12, height = 5)
# Plot layout and style
par(family = "Times", mfrow = c(1, 3), oma = c(2, 1, 2, 1),
    cex.axis = 1.5, cex.lab = 2, cex.main = 2.2)

for (j in 1:3) {
  par(mar = c(4, ifelse(j == 1, 5, 1), 3, 1))
  plot(NA, ylim = c(-3, 3), xlim = c(1, num_time),
       xlab = "measurement occasion t",
       ylab = ifelse(j == 1, "outcome y", ""),
       main = paste0("BAI ", j), axes = FALSE)
  axis(1)
  axis(2)
  # Plot all but Person 8 in different colors
  for (i in setdiff(seq_len(num_obs), 8)) {
    lines(y_data[i, , j], col = i, lwd = 1)
  }
  # Draw Person 1 last for emphasis
  lines(y_data[8, , j], col = 1, lwd = 4)
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

# SOME INTERPRETATION
# The range of starting values varies
# -> indicates the necessity of a random intercept
# The lines develop similarly and appear roughly parallel, especially for BAI 1
# -> less likely that a random slope is needed


################################################################################
# MODEL 1: 2-level single item AR(1) with person-specific random
#          intercept and slope
################################################################################

# Define parameters which are being estimated
params_model1 <- c("alpha", "beta", "sigma.y", "sigma.eps2", "rho.eps2")
# alpha: first AR(1) parameter (random intercept/baseline value)
# beta: second AR(1) parameter (slope/autoregressive coefficient)
# sigma.y: variance of the residuals
# sigma.eta2: variance of the random effects
# rho.eta2: correlation matrix  of the random effects

# Run jags model for item 10 (1st BAI indicator)
# with pre-selected burn-in (see convergence check below, APPENDIX 3.0)
model1 <- jags.parallel(
  data = data_prepared, # data to use
  parameters.to.save = params_model1, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("03_mlm/models",
                          "mlm1_observed_ar1_2level-intslop.txt")
)

# Store parameter estimates
est1 <- model1$BUGSoutput$summary
# Print parameter estimates
round(est1, 2)
#-------------------------------------------------------------------------------
# Inference for Bugs model "mlm1_observed_ar1_2level-intslop.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
#                  mean    sd    2.5%     25%     50%     75%   97.5% Rhat n.eff
# alpha           -0.42  0.09   -0.58   -0.48   -0.42   -0.36   -0.24 1.01   370
# beta             0.45  0.06    0.32    0.40    0.45    0.49    0.57 1.00   720
# rho.eps2[1,1]    1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# rho.eps2[2,1]    0.17  0.19   -0.22    0.04    0.18    0.31    0.53 1.00   980
# rho.eps2[1,2]    0.17  0.19   -0.22    0.04    0.18    0.31    0.53 1.00   980
# rho.eps2[2,2]    1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# sigma.eps2[1,1]  0.33  0.08    0.20    0.27    0.32    0.38    0.52 1.00  1600
# sigma.eps2[2,1]  0.03  0.04   -0.04    0.01    0.03    0.06    0.12 1.00   860
# sigma.eps2[1,2]  0.03  0.04   -0.04    0.01    0.03    0.06    0.12 1.00   860
# sigma.eps2[2,2]  0.11  0.03    0.06    0.09    0.10    0.12    0.18 1.00 12000
# sigma.y          0.28  0.02    0.25    0.27    0.28    0.29    0.31 1.00  2900
#
# SOME INTERPRETATION
# - alpha: significant negative overall baseline level across individuals
# - beta: significant postive temporal dependency across individuals
# - sigma.eps2[1,1]: moderate variability in individual baseline levels
# - sigma.eps2[2,2]: small variability in individual trends
# - rho.eps2[2,1]: correlation between random slope and intercept close to zero
# (random effect could be modeled independently rather than jointly distributed)
# - sigma.y: samll within-individual variability not explained by the model,
#   suggesting good model fit
#-------------------------------------------------------------------------------


################################################################################
# MODEL 2: 2-level single item AR(1) with person-specific random
#          intercept and slope and time-specific random slope
################################################################################

# Define parameters which are being estimated
params_model2 <- c(
  "alpha", "beta", "sigma.y", "sigma.eps2",
  "rho.eps2", "sigma.eps3"
)
# alpha: first AR(1) parameter (random intercept/baseline value)
# beta: second AR(1) parameter (slope/autoregressive coefficient)
# sigma.y: variance of the residuals
# sigma.eps2: variance of the random effects
# rho.eps2: correlation matrix  of the random effects
# sigma.eps3: variance of the time-specific random slope

# Run jags model (check of convergence below)
model2 <- jags.parallel(
  data = data_prepared, # data to use
  parameters.to.save = params_model2, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter,
  model.file =  here::here("03_mlm/models",
                          "mlm2_observed_ar1_cross.txt")
)

# Store parameter estimates
est2 <- model2$BUGSoutput$summary
# Print parameter estimates
round(est2, 2)
#-------------------------------------------------------------------------------
# Inference for Bugs model "lm2_observed_ar1_cross.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
#                  mean    sd    2.5%     25%     50%     75%   97.5% Rhat n.eff
# alpha           -0.41  0.09   -0.58   -0.47   -0.41   -0.35   -0.23 1.01   560
# beta             0.44  0.14    0.16    0.35    0.44    0.53    0.70 1.00   780
# rho.eps2[1,1]    1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# rho.eps2[2,1]    0.14  0.20   -0.25    0.01    0.15    0.28    0.50 1.00   980
# rho.eps2[1,2]    0.14  0.20   -0.25    0.01    0.15    0.28    0.50 1.00   980
# rho.eps2[2,2]    1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# sigma.eps2[1,1]  0.35  0.09    0.21    0.29    0.34    0.40    0.55 1.00   980
# sigma.eps2[2,1]  0.03  0.04   -0.05    0.00    0.03    0.05    0.11 1.00  1000
# sigma.eps2[1,2]  0.03  0.04   -0.05    0.00    0.03    0.05    0.11 1.00  1000
# sigma.eps2[2,2]  0.11  0.03    0.06    0.09    0.10    0.13    0.18 1.00  9000
# sigma.eps3       0.20  0.09    0.09    0.14    0.18    0.24    0.41 1.00 16000
# sigma.y          0.27  0.01    0.24    0.26    0.27    0.28    0.30 1.00  2700
#
# SOME INTERPRETATION
# - alpha, beta, sigma.eps2, rho.eps2 and sigma.y take similar values as
#   in previous 2-level model
# - sigma.zeta3 estimated to be greater than zero
#-------------------------------------------------------------------------------


################################################################################
# CONVERGENCE chekcs
################################################################################

############################ Convergence Model 1 ###############################

# Run jags model 1 with burn-in parameter set to 1
model1_conv <- jags.parallel(
  data = data_prepared, # data to use (t=1)
  parameters.to.save = params_model1, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file =  here::here("03_mlm/models",
                           "mlm1_observed_ar1_2level-intslop.txt")
)
# file containing the model

# Create Markov Chain Monte Carlo samples
samps1 <- as.mcmc(model1_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("03_mlm/convergence_checks",
               "mlm_model1_gelmanplot.pdf"))
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
pdf(file = here::here("03_mlm/convergence_checks",
                      "mlm_model1_xyplot.pdf"), pointsize = 6)
plot(samps1)
# When the model is converged after the burn-in phase,
# the trace should not exhibit any increasing or decreasing trends,
# the different chains (plotted in different colors) should mix well
# and the density plot should be smooth and have only a single peak.
dev.off() # reset layout

# -> The model seems to be converged after 5000 MCMC iterations.

############################ Convergence Model 2 ###############################

# Run jags model at t=10 with burn-in parameter set to 1
model2_conv <- jags.parallel(
  data = data_prepared, # data to use (t=10)
  parameters.to.save = params_model2, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file =  here::here("03_mlm/models",
                           "mlm2_observed_ar1_cross.txt")
)
# file containing the model

# Create Markov Chain Monte Carlo samples
samps2 <- as.mcmc(model2_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("03_mlm/convergence_checks",
               "mlm_model2_gelmanplot.pdf"))
par(new = F)
for (v in 1:nvar(samps2)) {
  # Plot potential scale reduction factors against MCMC iterations
  gelman.plot(samps2[, v], ylim = c(0, 7), col = c("black", NA))
  par(new = T)
}
# Draw horizontal reference line at 1.1
abline(h = 1.1)
dev.off() # reset layout

# Create trace plots and density plots for each parameter
pdf(file = here::here("03_mlm/convergence_checks",
                      "mlm_model2_xyplot.pdf"), pointsize = 6)
plot(samps2)
dev.off() # reset layout

# -> The model seems to be converged after 5000 MCMC iterations.


################################################################################
# APPENDIX MODEL 1a: ARMA(1,1) Time Series Structure
################################################################################

# Define parameters which are being estimated
params_model1a <- c("alpha", "beta", "gamma", "sigma.y",
                    "sigma.eps2", "rho.eps2")
# alpha: first AR(1) parameter (random intercept/baseline value)
# beta: second AR(1) parameter (slope/autoregressive coefficient)
# gamma: MA(1) parameter
# sigma.y: variance of the residuals
# sigma.eta2: variance of the random effects
# rho.eta2: correlation matrix  of the random effects

# Run jags model for item 1 (1st BAI indicator)
# with pre-selected burn-in (see convergence check below, APPENDIX 3.0)
model1a <- jags.parallel(
  data = data_prepared, # data to use
  parameters.to.save = params_model1a, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("03_mlm/models/appendix",
                          "mlm1a_observed_ar1ma1_2level-intslop.txt")
)

# Store parameter estimates
est1a <- model1a$BUGSoutput$summary
# Print parameter estimates
round(est1a, 2)
#-------------------------------------------------------------------------------
# Inference for Bugs model "mlm1a_observed_ar1ma1_2level-intslop.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
#                  mean    sd    2.5%     25%     50%     75%   97.5% Rhat n.eff
# alpha           -0.42  0.09   -0.59   -0.48   -0.43   -0.37   -0.25 1.03   100
# beta             0.42  0.12    0.18    0.35    0.43    0.50    0.64 1.01   240
# gamma            0.01  0.11   -0.20   -0.06    0.01    0.08    0.23 1.00   840
# rho.eps2[1,1]    1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# rho.eps2[2,1]    0.18  0.20   -0.22    0.05    0.19    0.32    0.54 1.01   570
# rho.eps2[1,2]    0.18  0.20   -0.22    0.05    0.19    0.32    0.54 1.01   570
# rho.eps2[2,2]    1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# sigma.eps2[1,1]  0.33  0.08    0.20    0.27    0.32    0.38    0.52 1.00  7600
# sigma.eps2[2,1]  0.04  0.04   -0.04    0.01    0.03    0.06    0.12 1.00   750
# sigma.eps2[1,2]  0.04  0.04   -0.04    0.01    0.03    0.06    0.12 1.00   750
# sigma.eps2[2,2]  0.11  0.03    0.06    0.09    0.10    0.13    0.19 1.00   640
# sigma.y          0.28  0.02    0.25    0.27    0.28    0.29    0.31 1.00  4300
#
# SOME INTERPRETATION
# - additional MA parameter gamma close to zero
# -> indicates that the ARMA(1,1) model is very similar to the AR(1) model
#-------------------------------------------------------------------------------
