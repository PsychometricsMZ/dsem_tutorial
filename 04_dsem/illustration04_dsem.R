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
# MODEL 1: One-factor AR(1) with person-specific random intercept and slope
# -> dsem_model1_1factor_ar1_2level-intslop.txt
#
# MODEL 2: One-factor AR(1) with person-specific intercept + slope
#          and time-specific random slope
# -> dsem_model1_1factor_ar1_cross.txt
#
# CONVERGENCE checks
#
# APPENDIX MODEL 1a: Four-factor AR(1) with person-specific random intercept
# -> dsem_model2_4factors_ar1_2level-int.txt
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
# MODEL 1:  One-factor AR(1) with person-specific random intercept
#           and slope
################################################################################

# Define parameters which are being estimated
params_model1 <- c(
  "alpha", "beta", "lambda.y", "nu.y", "sigma.y", "sigma.eta",
  "sigma.zeta2", "rho.zeta2"
)
# alpha: first AR(1) parameter (random intercept/baseline value)
# beta: second AR(1) parameter (slope/autoregressive coefficient)
# lambda.y: factor loadings
# lambda.y.strd: standardized factor loadings
# nu.y: factor intercepts
# sigma.y: variance of the residuals
# sigma.eta: variance of the latent factor
# sigma.zeta2: variance of the random effects
# rho.zeta2: correlation matrix  of the random effects

# Run jags model for items 10-12 (all BAI indicators)
model1 <- jags.parallel(
  data = data_1factor, # data to use
  parameters.to.save = params_model1, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("04_dsem/models",
                          "dsem_model1_1factor_ar1_2level-intslop.txt")
)

# Store parameter estimates
est1 <- model1$BUGSoutput$summary
# Print parameter estimates
round(est1, 2)
#-------------------------------------------------------------------------------
# Inference for Bugs model "dsem_model1_1factor_ar1_2level-intslop.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
#                  mean    sd    2.5%     25%     50%     75%   97.5% Rhat n.eff
# alpha           -0.29  0.10   -0.49   -0.36   -0.29   -0.23   -0.10 1.01  1000
# beta             0.71  0.07    0.58    0.66    0.71    0.75    0.84 1.02   130
# lambda.y[1]      0.89  0.05    0.79    0.86    0.89    0.93    0.99 1.00  1300
# lambda.y[2]      1.36  0.07    1.23    1.31    1.36    1.40    1.51 1.01   300
# nu.y[1]          0.05  0.04   -0.03    0.02    0.04    0.07    0.12 1.00  1400
# nu.y[2]          0.10  0.05    0.01    0.07    0.10    0.13    0.20 1.01   310
# rho.zeta2[1,1]   1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# rho.zeta2[2,1]   0.52  0.13    0.23    0.44    0.53    0.62    0.74 1.00  1100
# rho.zeta2[1,2]   0.52  0.13    0.23    0.44    0.53    0.62    0.74 1.00  1100
# rho.zeta2[2,2]   1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# sigma.eta        0.07  0.01    0.06    0.07    0.07    0.08    0.09 1.01   540
# sigma.y[1]       0.28  0.02    0.25    0.27    0.28    0.30    0.33 1.00  1500
# sigma.y[2]       0.33  0.02    0.29    0.32    0.33    0.35    0.38 1.00 12000
# sigma.y[3]       0.26  0.03    0.19    0.23    0.26    0.28    0.32 1.01   960
# sigma.zeta2[1,1] 0.40  0.10    0.24    0.32    0.38    0.45    0.63 1.01   710
# sigma.zeta2[2,1] 0.10  0.04    0.03    0.07    0.10    0.12    0.19 1.00   640
# sigma.zeta2[1,2] 0.10  0.04    0.03    0.07    0.10    0.12    0.19 1.00   640
# sigma.zeta2[2,2] 0.09  0.03    0.05    0.07    0.09    0.11    0.16 1.01   280
#
# SOME INTERPRETATION
# - intercept alpha with credible interval below zero indicates slight
#   negative baseline value for the latent process
# - autoregressive coefficient beta credible interval above zero shows moderate
#   positive autoregressive dependence -> the current value of the latent
#   variable is positively influenced by its previous value
# - moderate to high factor loadings with credible intervals above zero
#   indicate robust relationships between indicators and latent variable with
#   minor baseline offsets according to the small intercept estimates
# - moderate variability in person-specific random effects with small
#   correlation between intercept and slope
#-------------------------------------------------------------------------------

# Print correlation matrix of the random effects
round(matrix(est1[c(
  paste0("rho.zeta2[", 1:2, ",1]"),
  paste0("rho.zeta2[", 1:2, ",2]")
), "mean"], 2, 2), 3)
#-------------------------------------------------------------------------------
# [,1]  [,2]
# [1,] 1.000 0.519
# [2,] 0.519 1.000
#
# SOME INTERPRETATION
# -> Positive correlation between random slope and intercept which implies
#    that people with a higher starting / baseline value experience
#    a sharper increase
#-------------------------------------------------------------------------------

# Calculate standardized factor loadings (see code in utils/standardization.R)
result_model1 <- run_factor_analysis_map(
  est = est1,
  factor_structure = factor_structure,
  n_factors = 1,
  person_intercept_var_name = c("sigma.zeta2[1,1]"),
  person_slope_var_name = c("sigma.zeta2[2,2]")
)
print_analysis_results(result_model1)
#-------------------------------------------------------------------------------
# Standardized Factor Loadings:
#   indicator_1 indicator_2 indicator_3
# 1       0.831       0.776       0.906
#
# Intraclass Correlation Coefficient (ICC):
# 0.686
#
# Total Latent Variance (Factor Level):
#   0.635
#   Splitted into:
#     Raw Dynamic (ARMA) Factor Variance:
#       0.146
#     Raw Person-Level Intercept Variance:
#       0.395
#     Raw Person-Level Slope Variance:
#       0.094
#
# SOME INTERPRETATION
# -> factor loadings look good
#-------------------------------------------------------------------------------


################################################################################
# MODEL 2: One-factor AR(1) with person-specific intercept + slope
#          and time-specific random slope
################################################################################

# Define parameters which are being estimated
params_model2 <- c(
  "alpha", "beta", "lambda.y", "nu.y", "sigma.y", "sigma.eta",
  "sigma.zeta2", "rho.zeta2", "sigma.zeta3"
)
# alpha: first AR(1) parameter (random intercept/baseline value)
# beta: second AR(1) parameter (slope/autoregressive coefficient)
# lambda.y: factor loadings
# lambda.y.strd: standardized factor loadings
# nu.y: factor intercepts
# sigma.y: variance of the residuals
# sigma.eta: variance of the latent factor
# sigma.zeta2: variance of the person-specific random effects
# rho.zeta2: correlation matrix  of the random effects
# sigma.zeta3: variance of the time-specific random effects

# Run jags model for items 10-12 (all BAI indicators)
model2 <- jags.parallel(
  data = data_1factor, # data to use
  parameters.to.save = params_model2, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("04_dsem/models",
                          "dsem_model1_1factor_ar1_cross.txt")
)

# Store parameter estimates
est2 <- model2$BUGSoutput$summary
# Print parameter estimates
round(est2, 2)
#-------------------------------------------------------------------------------
# Inference for Bugs model "dsem_model1_1factor_ar1_cross.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#                  mean    sd    2.5%     25%     50%     75%   97.5% Rhat n.eff
# alpha           -0.39  0.10   -0.58   -0.46   -0.40   -0.33   -0.19 1.05    59
# beta             0.65  0.15    0.35    0.55    0.65    0.75    0.92 1.06    49
# lambda.y[1]      0.90  0.05    0.80    0.86    0.90    0.93    1.00 1.00  1500
# lambda.y[2]      1.34  0.07    1.21    1.29    1.33    1.38    1.47 1.01   260
# nu.y[1]          0.05  0.04   -0.02    0.02    0.05    0.07    0.12 1.00  9600
# nu.y[2]          0.09  0.05    0.01    0.06    0.09    0.12    0.18 1.00  1100
# rho.zeta2[1,1]   1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# rho.zeta2[2,1]   0.54  0.13    0.24    0.46    0.56    0.64    0.76 1.00  1200
# rho.zeta2[1,2]   0.54  0.13    0.24    0.46    0.56    0.64    0.76 1.00  1200
# rho.zeta2[2,2]   1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# sigma.eta        0.07  0.01    0.05    0.06    0.06    0.07    0.08 1.00   870
# sigma.y[1]       0.28  0.02    0.25    0.27    0.28    0.30    0.32 1.01   360
# sigma.y[2]       0.33  0.02    0.28    0.31    0.33    0.34    0.38 1.00  5100
# sigma.y[3]       0.27  0.03    0.21    0.25    0.27    0.29    0.34 1.01   360
# sigma.zeta2[1,1] 0.42  0.11    0.24    0.34    0.40    0.48    0.68 1.00   650
# sigma.zeta2[2,1] 0.12  0.05    0.04    0.09    0.12    0.15    0.23 1.01   630
# sigma.zeta2[1,2] 0.12  0.05    0.04    0.09    0.12    0.15    0.23 1.01   630
# sigma.zeta2[2,2] 0.12  0.04    0.06    0.09    0.11    0.14    0.21 1.01   960
# sigma.zeta3      0.22  0.10    0.10    0.16    0.20    0.27    0.49 1.00  9100
#-------------------------------------------------------------------------------

# Calculate standardized factor loadings (see code in utils/standardization.R)
result_model2 <- run_factor_analysis_map(
  est = est2,
  factor_structure = factor_structure,
  n_factors = 1,
  person_intercept_var_name = c("sigma.zeta2[1,1]"),
  person_slope_var_name = c("sigma.zeta2[2,2]"),
  time_slope_var_name = c("sigma.zeta3")
)
print_analysis_results(result_model2)
#-------------------------------------------------------------------------------
# Standardized Factor Loadings:
#   indicator_1 indicator_2 indicator_3
# 1       0.869       0.825       0.923
#
# Intraclass Correlation Coefficient (ICC):
# 0.748
#
# Total Latent Variance (Factor Level):
#   0.872
#   Splitted into:
#     Raw Dynamic (ARMA) Factor Variance:
#       0.113
#     Raw Person-Level Intercept Variance:
#       0.418
#     Raw Person-Level Slope Variance:
#       0.116
#
# SOME INTERPRETATION
# -> factor loadings look good
#-------------------------------------------------------------------------------


######################## Plot time-specific residuals ##########################

# Re-run model to withdraw time-specific residuals
model2_zeta3 <- jags.parallel(
  data = data_1factor, # data to use
  parameters.to.save = c(zeta3), # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("04_dsem/models",
                          "dsem_model1_1factor_ar1_cross.txt")
)

# Extract posterior summaries from model output
names_z3  <- paste0("zeta3[", 2:num_time, "]")
z3_hat <- model2_zeta3$BUGSoutput$summary[names_z3, "mean"]
pdf(here::here("04_dsem/plots", "timeplot.pdf"), width = 6, height = 6)
scatter.smooth(z3_hat, axes = FALSE,
               xlab = "Session", ylab = expression(zeta[3]))
axis(1)
axis(2)
dev.off()


################################################################################
# CONVERGENCE checks
################################################################################

############################ Convergence Model 1 ###############################

# Run jags model 1 with burn-in parameter set to 1
model1_conv <- jags.parallel(
  data = data_1factor, # data to use (t=1)
  parameters.to.save = params_model1, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file =  here::here("04_dsem/models",
                           "dsem_model1_1factor_ar1_2level-intslop.txt")
)

# Create Markov Chain Monte Carlo samples
samps1 <- as.mcmc(model1_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("04_dsem/convergence_checks",
               "dsem_model1_gelmanplot.pdf"))
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
pdf(file = here::here("04_dsem/convergence_checks",
                      "dsem_model1_xyplot.pdf"), pointsize = 6)
plot(samps1)
# When the model is converged after the burn-in phase,
# the trace should not exhibit any increasing or decreasing trends,
# the different chains (plotted in different colors) should mix well
# and the density plot should be smooth and have only a single peak.
dev.off() # reset layout

# -> The model seems to be converged after 5000 MCMC iterations.

############################ Convergence Model 2 ###############################

# Run jags model 2 with burn-in parameter set to 1
model2_conv <- jags.parallel(
  data = data_1factor, # data to use (t=1)
  parameters.to.save = params_model2, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file =  here::here("04_dsem/models",
                           "dsem_model1_1factor_ar1_cross.txt")
)

# Create Markov Chain Monte Carlo samples
samps2 <- as.mcmc(model2_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("04_dsem/convergence_checks",
               "dsem_model2_gelmanplot.pdf"))
par(new = FALSE)
# Loop through each parameter in parameters.to.save
for (v in 1:nvar(samps2)) {
  # Plot potential scale reduction factors against MCMC iterations
  gelman.plot(samps2[, v], ylim = c(0, 7), col = c("black", NA))
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
pdf(file = here::here("04_dsem/convergence_checks",
                      "dsem_model2_xyplot.pdf"), pointsize = 6)
plot(samps2)
# When the model is converged after the burn-in phase,
# the trace should not exhibit any increasing or decreasing trends,
# the different chains (plotted in different colors) should mix well
# and the density plot should be smooth and have only a single peak.
dev.off() # reset layout

# -> The model seems to be converged after 5000 MCMC iterations.


################################################################################
# APPENDIX MODEL 2:  Four-factor AR(1) with person-specific random intercept
################################################################################

# Prepare data for four factors
data_4factors <- list(
  N = num_obs, # sample size
  Nt = num_time, # time points
  y = y_data, # repsonses to item 1
  psi0 = diag(4), # hyperprior wishart distribution
  mu.zeta21 = rep(0, 4)  # hyperprior multivariate normal distribution
)

# Define parameters which are being estimated
params_model3 <- c(
  "alpha", "beta", "lambda.y", "lambda.y.strd", "nu.y",
  "sigma.y", "sigma.eta", "sigma.zeta21", "rho.zeta21"
)
# alpha: first AR(1) parameter (random intercept/baseline value)
# beta: second AR(1) parameter (slope/autoregressive coefficient)
# lambda.y: factor loadings
# lambda.y.strd: strd factor loadings
# nu.y: factor intercepts
# sigma.y: variance of the residuals
# sigma.eta: variance of the latent factor
# sigma.zeta21: variance of the random intercepts
# rho.zeta21: correlation matrix  of the random intercepts

# Run jags model for all items with burn-in parameter set to 5000
model3 <- jags.parallel(
  data = data_4factors, # data to use
  parameters.to.save = params_model3, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("04_dsem/models/appendix",
                          "dsem_model2_4factors_ar1_2level-int.txt")
)

# Store parameter estimates
est3 <- model3$BUGSoutput$summary
# Print parameter estimates
round(est3, 2)
#-------------------------------------------------------------------------------
# Inference for Bugs model "dsem_model2_4factors_ar1_2level-int.txt",
# fit using jags, 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
#                     mean    sd   2.5%   25%    50%   75%    97.5% Rhat n.eff
# alpha[1]           -0.09  0.09  -0.27  -0.16  -0.10  -0.03   0.08 1.00  4400
# alpha[2]            0.15  0.08   0.00   0.10   0.15   0.20   0.30 1.00  8100
# alpha[3]            0.30  0.09   0.13   0.25   0.31   0.36   0.47 1.01   240
# alpha[4]            0.01  0.10  -0.17  -0.05   0.01   0.08   0.20 1.00  2200
# beta[1,1]           0.87  0.05   0.75   0.83   0.87   0.91   0.96 1.05  64
# beta[2,1]          -0.20  0.07  -0.34  -0.25  -0.20  -0.16  -0.06 1.03  110
# beta[3,1]          -0.34  0.08  -0.49  -0.39  -0.34  -0.28  -0.18 1.03  86
# beta[4,1]          -0.26  0.07  -0.41  -0.31  -0.26  -0.22  -0.13 1.02  210
# beta[1,2]           0.26  0.15  -0.03   0.16   0.26   0.36   0.54 1.10  32
# beta[2,2]           0.13  0.18  -0.25   0.01   0.13   0.25   0.46 1.15  25
# beta[3,2]          -0.29  0.23  -0.77  -0.45  -0.30  -0.15   0.19 1.16  22
# beta[4,2]          -0.03  0.18  -0.35  -0.16  -0.04   0.09   0.35 1.07  53
# beta[1,3]          -0.10  0.12  -0.31  -0.19  -0.10  -0.03   0.16 1.05  64
# beta[2,3]           0.09  0.13  -0.17   0.00   0.09   0.18   0.33 1.07  46
# beta[3,3]           0.42  0.19   0.05   0.29   0.42   0.56   0.78 1.16  21
# beta[4,3]           0.22  0.14  -0.04   0.13   0.22   0.31   0.49 1.02   170
# beta[1,4]          -0.27  0.15  -0.57  -0.37  -0.27  -0.18   0.02 1.06  72
# beta[2,4]           0.20  0.17  -0.14   0.09   0.21   0.33   0.50 1.20  21
# beta[3,4]           0.38  0.21  -0.10   0.26   0.40   0.53   0.73 1.16  29
# beta[4,4]           0.38  0.19  -0.01   0.24   0.38   0.51   0.73 1.10  41
# lambda.y[1]         0.93  0.05   0.82   0.89   0.93   0.96   1.04 1.00 14000
# lambda.y[2]         1.38  0.07   1.24   1.33   1.37   1.42   1.52 1.00   620
# lambda.y[3]         1.13  0.05   1.04   1.10   1.13   1.16   1.23 1.00  2200
# lambda.y[4]         1.10  0.05   1.01   1.07   1.10   1.14   1.20 1.00  1400
# lambda.y[5]         0.95  0.04   0.87   0.92   0.95   0.98   1.04 1.01   410
# lambda.y[6]         0.83  0.05   0.74   0.80   0.83   0.86   0.92 1.00  3000
# lambda.y[7]         1.06  0.04   0.97   1.03   1.05   1.08   1.14 1.00  1600
# lambda.y[8]         0.81  0.05   0.72   0.78   0.81   0.84   0.91 1.00  7900
# lambda.y.strd[1]    0.38  0.03   0.32   0.36   0.38   0.40   0.44 1.00  4400
# lambda.y.strd[2]    0.34  0.03   0.28   0.32   0.34   0.35   0.39 1.00  2300
# lambda.y.strd[3]    0.50  0.04   0.42   0.47   0.50   0.53   0.59 1.00   940
# lambda.y.strd[4]    0.53  0.02   0.48   0.51   0.53   0.54   0.57 1.01   500
# lambda.y.strd[5]    0.63  0.03   0.57   0.61   0.63   0.65   0.68 1.01   400
# lambda.y.strd[6]    0.59  0.03   0.54   0.57   0.59   0.61   0.64 1.01   530
# lambda.y.strd[7]    0.51  0.03   0.45   0.48   0.51   0.53   0.56 1.01   190
# lambda.y.strd[8]    0.57  0.04   0.50   0.55   0.57   0.59   0.64 1.03  98
# lambda.y.strd[9]    0.43  0.03   0.37   0.41   0.43   0.46   0.49 1.02   170
# lambda.y.strd[10]   0.45  0.03   0.39   0.43   0.45   0.46   0.50 1.01   590
# lambda.y.strd[11]   0.56  0.04   0.49   0.54   0.56   0.59   0.63 1.00   740
# lambda.y.strd[12]   0.33  0.03   0.28   0.31   0.33   0.35   0.38 1.00   880
# nu.y[1]             0.06  0.04  -0.01   0.04   0.06   0.09   0.14 1.00 20000
# nu.y[2]             0.11  0.05   0.02   0.08   0.11   0.14   0.20 1.00   720
# nu.y[3]            -0.14  0.03  -0.21  -0.17  -0.14  -0.12  -0.08 1.00  5300
# nu.y[4]            -0.11  0.03  -0.18  -0.14  -0.11  -0.09  -0.05 1.00  8600
# nu.y[5]            -0.47  0.04  -0.55  -0.50  -0.47  -0.45  -0.40 1.01   490
# nu.y[6]            -0.38  0.04  -0.46  -0.41  -0.38  -0.35  -0.30 1.00  3800
# nu.y[7]             0.15  0.03   0.08   0.13   0.15   0.17   0.21 1.00   820
# nu.y[8]             0.38  0.04   0.31   0.35   0.38   0.40   0.45 1.00  1600
# rho.zeta21[1,1]     1.00  0.00   1.00   1.00   1.00   1.00   1.00 1.00   1
# rho.zeta21[2,1]     0.04  0.15  -0.26  -0.07   0.04   0.15   0.33 1.00 20000
# rho.zeta21[3,1]     0.01  0.16  -0.30  -0.10   0.01   0.12   0.32 1.00  1100
# rho.zeta21[4,1]    -0.03  0.16  -0.33  -0.14  -0.03   0.08   0.27 1.00  4300
# rho.zeta21[1,2]     0.04  0.15  -0.26  -0.07   0.04   0.15   0.33 1.00 20000
# rho.zeta21[2,2]     1.00  0.00   1.00   1.00   1.00   1.00   1.00 1.00   1
# rho.zeta21[3,2]     0.63  0.09   0.42   0.57   0.64   0.70   0.79 1.00 20000
# rho.zeta21[4,2]     0.70  0.08   0.53   0.66   0.71   0.76   0.83 1.00 11000
# rho.zeta21[1,3]     0.01  0.16  -0.30  -0.10   0.01   0.12   0.32 1.00  1100
# rho.zeta21[2,3]     0.63  0.09   0.42   0.57   0.64   0.70   0.79 1.00 20000
# rho.zeta21[3,3]     1.00  0.00   1.00   1.00   1.00   1.00   1.00 1.00   1
# rho.zeta21[4,3]     0.44  0.12   0.18   0.36   0.45   0.53   0.66 1.00 20000
# rho.zeta21[1,4]    -0.03  0.16  -0.33  -0.14  -0.03   0.08   0.27 1.00  4300
# rho.zeta21[2,4]     0.70  0.08   0.53   0.66   0.71   0.76   0.83 1.00 11000
# rho.zeta21[3,4]     0.44  0.12   0.18   0.36   0.45   0.53   0.66 1.00 20000
# rho.zeta21[4,4]     1.00  0.00   1.00   1.00   1.00   1.00   1.00 1.00   1
# sigma.eta[1,1]      0.05  0.01   0.04   0.04   0.05   0.05   0.06 1.00  4400
# sigma.eta[2,1]      0.00  0.01  -0.02  -0.01   0.00   0.00   0.01 1.00  1100
# sigma.eta[3,1]      0.00  0.01  -0.02  -0.01   0.00   0.00   0.01 1.02   170
# sigma.eta[4,1]      0.00  0.01  -0.01   0.00   0.00   0.00   0.01 1.00   760
# sigma.eta[1,2]      0.00  0.01  -0.02  -0.01   0.00   0.00   0.01 1.00  1100
# sigma.eta[2,2]      0.11  0.01   0.09   0.10   0.11   0.11   0.13 1.01   430
# sigma.eta[3,2]      0.07  0.01   0.06   0.07   0.07   0.08   0.09 1.00   750
# sigma.eta[4,2]      0.08  0.01   0.06   0.07   0.08   0.08   0.10 1.00  5200
# sigma.eta[1,3]      0.00  0.01  -0.02  -0.01   0.00   0.00   0.01 1.02   170
# sigma.eta[2,3]      0.07  0.01   0.06   0.07   0.07   0.08   0.09 1.00   750
# sigma.eta[3,3]      0.11  0.02   0.08   0.10   0.11   0.12   0.14 1.02   140
# sigma.eta[4,3]      0.07  0.01   0.05   0.06   0.06   0.07   0.09 1.01   240
# sigma.eta[1,4]      0.00  0.01  -0.01   0.00   0.00   0.00   0.01 1.00   760
# sigma.eta[2,4]      0.08  0.01   0.06   0.07   0.08   0.08   0.10 1.00  5200
# sigma.eta[3,4]      0.07  0.01   0.05   0.06   0.06   0.07   0.09 1.01   240
# sigma.eta[4,4]      0.10  0.01   0.08   0.09   0.10   0.11   0.13 1.01   470
# sigma.y[1]          0.29  0.02   0.26   0.28   0.29   0.31   0.33 1.00 11000
# sigma.y[2]          0.33  0.02   0.28   0.31   0.33   0.34   0.37 1.00  3300
# sigma.y[3]          0.27  0.03   0.22   0.25   0.27   0.29   0.33 1.00   640
# sigma.y[4]          0.28  0.02   0.24   0.26   0.28   0.29   0.31 1.00  2300
# sigma.y[5]          0.21  0.02   0.18   0.20   0.21   0.22   0.24 1.00  1100
# sigma.y[6]          0.24  0.02   0.21   0.23   0.24   0.25   0.27 1.00  3500
# sigma.y[7]          0.32  0.02   0.28   0.31   0.32   0.33   0.37 1.00 20000
# sigma.y[8]          0.21  0.02   0.18   0.20   0.21   0.22   0.24 1.01   350
# sigma.y[9]          0.33  0.02   0.29   0.31   0.33   0.34   0.37 1.00  3300
# sigma.y[10]         0.40  0.03   0.36   0.39   0.40   0.42   0.46 1.00  4200
# sigma.y[11]         0.24  0.02   0.20   0.23   0.24   0.25   0.28 1.00  3500
# sigma.y[12]         0.54  0.03   0.47   0.51   0.53   0.56   0.61 1.00  5800
# sigma.zeta21[1,1]   0.39  0.09   0.24   0.32   0.37   0.44   0.60 1.00  3400
# sigma.zeta21[2,1]   0.01  0.05  -0.09  -0.02   0.01   0.05   0.11 1.00 20000
# sigma.zeta21[3,1]   0.00  0.06  -0.12  -0.03   0.00   0.04   0.11 1.00  1200
# sigma.zeta21[4,1]  -0.01  0.07  -0.15  -0.05  -0.01   0.03   0.11 1.00  5500
# sigma.zeta21[1,2]   0.01  0.05  -0.09  -0.02   0.01   0.05   0.11 1.00 20000
# sigma.zeta21[2,2]   0.28  0.06   0.18   0.23   0.27   0.31   0.42 1.00  3600
# sigma.zeta21[3,2]   0.19  0.06   0.09   0.15   0.18   0.22   0.31 1.00 14000
# sigma.zeta21[4,2]   0.24  0.07   0.14   0.20   0.24   0.28   0.39 1.00  6400
# sigma.zeta21[1,3]   0.00  0.06  -0.12  -0.03   0.00   0.04   0.11 1.00  1200
# sigma.zeta21[2,3]   0.19  0.06   0.09   0.15   0.18   0.22   0.31 1.00 14000
# sigma.zeta21[3,3]   0.31  0.07   0.19   0.26   0.30   0.35   0.47 1.00  5000
# sigma.zeta21[4,3]   0.16  0.07   0.05   0.12   0.16   0.20   0.31 1.00  8600
# sigma.zeta21[1,4]  -0.01  0.07  -0.15  -0.05  -0.01   0.03   0.11 1.00  5500
# sigma.zeta21[2,4]   0.24  0.07   0.14   0.20   0.24   0.28   0.39 1.00  6400
# sigma.zeta21[3,4]   0.16  0.07   0.05   0.12   0.16   0.20   0.31 1.00  8600
# sigma.zeta21[4,4]   0.43  0.10   0.28   0.36   0.42   0.48   0.65 1.00  7800
#-------------------------------------------------------------------------------
