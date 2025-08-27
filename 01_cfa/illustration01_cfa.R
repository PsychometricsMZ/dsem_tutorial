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
# - MODEL: CFA for 4 factors with 3 indicators each
# -> cfa_model1_4factors.txt
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

######################## Prepare first time point t=1 ##########################

# Store measurement at time point t=1
y_data_t1 <- y_data[, 1, ]
# Print the  mean for each indicator, rounding and ignoring NA values
round(apply(y_data_t1, 2, mean, na.rm = TRUE), 3)
# SOME INTERPETATION: the data is centered around zero

# Prepare data at time point t=1 for jags
data_jags1 <- list(
  N = num_obs, # sample size
  y = y_data_t1, # data at time point t=1
  psi0 = diag(4) # diagonal matrix serving as hyperprior for factor covariance
)

####################### Prepare second time point t=15 #########################

# Store measurement at time point t=15
y_data_t15 <- y_data[, 15, ]
# Print the mean for each indicator, rounding and ignoring NA values
round(apply(y_data_t15, 2, mean, na.rm = TRUE), 3)
# SOME INTERPETATION: different means for the indicators

# Prepare data at time point t=1 for jags
data_jags15 <- list(
  N = num_obs, # sample size
  y = y_data_t15, # data at time point t=1
  psi0 = diag(4) # diagonal matrix serving as hyperprior for factor covariance
)


################################################################################
# MODEL: CFA for 4 factors with 3 indicators each
################################################################################

# Define parameters which are being estimated:
params <- c(
  "lambda.y", "nu.y", "sigma.eps", "sigma.eta", "rho.eta", "lambda.y.strd"
)
# lambda.y: free factor loadings
# nu.y: free factor intercepts
# sigma.eps: variance of the residuals
# sigma.eta: variance of the latent factor
# rho.eta: correlation matrix of the latent factors
# lambda.y.strd: standardized factor loadings (ranging from 0 to 1)

#################### Parameter estimates at time point t=1 #####################

# Run jags model with burn-in parameter set to 5000
model1 <- jags.parallel(
  data = data_jags1, # data to use (t=1)
  parameters.to.save = params, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("01_cfa/models", "cfa_model1_4factors.txt")
)
# file containaining the model
# Store parameter estimates
est1 <- model1$BUGSoutput$summary
# Print parameter estimates
round(est1, 2)
#-------------------------------------------------------------------------------
# Inference for Bugs model "cfa_model1_4factors.txt", fit using jags,
# 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
# The output contains the mean of the parameters, their standard deviation (sd),
# several quantiles of their posterior distribution in %,
# the Gelman-Rubin statistic / potential scale reduction factor Rhat
# and a crude measure of effective sample size n.eff.
#                  mean  sd      2.5%    25%     50%     75%    97.5% Rhat n.eff
# lambda.y[1]      1.02  0.28    0.53    0.82    0.99    1.18    1.64 1.00  7400
# lambda.y[2]      1.24  0.27    0.78    1.05    1.21    1.39    1.85 1.00 10000
# lambda.y[3]      1.34  0.25    0.92    1.16    1.31    1.49    1.88 1.01   530
# lambda.y[4]      1.35  0.27    0.89    1.16    1.33    1.51    1.94 1.01   510
# lambda.y[5]      1.29  0.32    0.74    1.06    1.26    1.48    1.98 1.00  1900
# lambda.y[6]      0.95  0.32    0.39    0.74    0.93    1.15    1.63 1.00  2300
# lambda.y[7]      1.29  0.27    0.81    1.10    1.27    1.45    1.87 1.00  1600
# lambda.y[8]      1.03  0.34    0.42    0.79    1.00    1.24    1.77 1.00  5100
# lambda.y.strd[1] 0.69  0.09    0.48    0.63    0.69    0.76    0.85 1.00  6600
# lambda.y.strd[2] 0.66  0.11    0.40    0.59    0.67    0.74    0.84 1.00  7000
# lambda.y.strd[3] 0.80  0.08    0.61    0.76    0.82    0.86    0.92 1.00 20000
# lambda.y.strd[4] 0.65  0.08    0.48    0.59    0.65    0.71    0.79 1.00   860
# lambda.y.strd[5] 0.86  0.05    0.73    0.83    0.87    0.90    0.94 1.00  5000
# lambda.y.strd[6] 0.83  0.06    0.68    0.80    0.84    0.88    0.93 1.00  2300
# lambda.y.strd[7] 0.60  0.09    0.42    0.54    0.60    0.66    0.77 1.00  2000
# lambda.y.strd[8] 0.74  0.10    0.51    0.68    0.76    0.82    0.90 1.00  5100
# lambda.y.strd[9] 0.55  0.13    0.26    0.47    0.56    0.64    0.77 1.00  7100
# lambda.y.strd[10]0.62  0.09    0.44    0.56    0.62    0.68    0.78 1.00  1900
# lambda.y.strd[11]0.78  0.10    0.55    0.73    0.80    0.85    0.92 1.00  3700
# lambda.y.strd[12]0.61  0.15    0.29    0.52    0.62    0.72    0.86 1.00  6000
# rho.eta[1,1]     1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# rho.eta[2,1]    -0.04  0.19   -0.40   -0.17   -0.04    0.09    0.33 1.00  7300
# rho.eta[3,1]    -0.15  0.20   -0.51   -0.30   -0.16   -0.02    0.25 1.00  3900
# rho.eta[4,1]    -0.21  0.19   -0.56   -0.34   -0.22   -0.08    0.17 1.00  1700
# rho.eta[1,2]    -0.04  0.19   -0.40   -0.17   -0.04    0.09    0.33 1.00  7300
# rho.eta[2,2]     1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# rho.eta[3,2]     0.64  0.12    0.37    0.57    0.66    0.73    0.84 1.00  7000
# rho.eta[4,2]     0.60  0.14    0.30    0.51    0.61    0.70    0.82 1.00  3400
# rho.eta[1,3]    -0.15  0.20   -0.51   -0.30   -0.16   -0.02    0.25 1.00  3900
# rho.eta[2,3]     0.64  0.12    0.37    0.57    0.66    0.73    0.84 1.00  7000
# rho.eta[3,3]     1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# rho.eta[4,3]     0.49  0.15    0.15    0.39    0.50    0.60    0.75 1.00  9400
# rho.eta[1,4]    -0.21  0.19   -0.56   -0.34   -0.22   -0.08    0.17 1.00  1700
# rho.eta[2,4]     0.60  0.14    0.30    0.51    0.61    0.70    0.82 1.00  3400
# rho.eta[3,4]     0.49  0.15    0.15    0.39    0.50    0.60    0.75 1.00  9400
# rho.eta[4,4]     1.00  0.00    1.00    1.00    1.00    1.00    1.00 1.00     1
# sigma.eps[1]     0.53  0.15    0.29    0.43    0.51    0.62    0.87 1.00  8500
# sigma.eps[2]     0.61  0.17    0.33    0.49    0.59    0.70    0.98 1.00  2700
# sigma.eps[3]     0.37  0.13    0.17    0.27    0.35    0.44    0.67 1.00  9100
# sigma.eps[4]     0.64  0.15    0.41    0.54    0.63    0.73    0.98 1.00 14000
# sigma.eps[5]     0.27  0.09    0.14    0.21    0.26    0.32    0.48 1.00  3700
# sigma.eps[6]     0.35  0.11    0.18    0.27    0.33    0.41    0.60 1.00  2400
# sigma.eps[7]     0.70  0.17    0.42    0.58    0.68    0.80    1.09 1.00  6700
# sigma.eps[8]     0.48  0.16    0.22    0.37    0.46    0.57    0.84 1.00  3400
# sigma.eps[9]     0.75  0.18    0.45    0.62    0.72    0.85    1.16 1.00 16000
# sigma.eps[10]    0.67  0.18    0.40    0.55    0.65    0.77    1.08 1.00  3900
# sigma.eps[11]    0.41  0.15    0.18    0.30    0.38    0.49    0.76 1.00  3200
# sigma.eps[12]    0.66  0.21    0.31    0.52    0.65    0.79    1.14 1.00 20000
# sigma.eta[1,1]   0.51  0.19    0.21    0.37    0.48    0.61    0.97 1.00  6300
# sigma.eta[2,1]  -0.02  0.10   -0.22   -0.08   -0.02    0.04    0.18 1.00  9800
# sigma.eta[3,1]  -0.07  0.10   -0.27   -0.13   -0.06   -0.01    0.12 1.00  2900
# sigma.eta[4,1]  -0.10  0.10   -0.32   -0.15   -0.09   -0.03    0.08 1.00  1200
# sigma.eta[1,2]  -0.02  0.10   -0.22   -0.08   -0.02    0.04    0.18 1.00  9800
# sigma.eta[2,2]   0.49  0.17    0.23    0.36    0.46    0.59    0.90 1.00   670
# sigma.eta[3,2]   0.29  0.12    0.11    0.20    0.27    0.35    0.56 1.00  2400
# sigma.eta[4,2]   0.27  0.11    0.10    0.19    0.26    0.33    0.54 1.00  1200
# sigma.eta[1,3]  -0.07  0.10   -0.27   -0.13   -0.06   -0.01    0.12 1.00  2900
# sigma.eta[2,3]   0.29  0.12    0.11    0.20    0.27    0.35    0.56 1.00  2400
# sigma.eta[3,3]   0.41  0.16    0.18    0.30    0.39    0.50    0.78 1.00  2100
# sigma.eta[4,3]   0.20  0.10    0.05    0.14    0.19    0.26    0.43 1.00  4200
# sigma.eta[1,4]  -0.10  0.10   -0.32   -0.15   -0.09   -0.03    0.08 1.00  1200
# sigma.eta[2,4]   0.27  0.11    0.10    0.19    0.26    0.33    0.54 1.00  1200
# sigma.eta[3,4]   0.20  0.10    0.05    0.14    0.19    0.26    0.43 1.00  4200
# sigma.eta[4,4]   0.43  0.16    0.20    0.32    0.41    0.52    0.82 1.00  1400
# nu.y[1]          0.02  0.16   -0.30   -0.08    0.02    0.13    0.34 1.00  1300
# nu.y[2]          0.00  0.16   -0.32   -0.10    0.00    0.11    0.33 1.00  1300
# nu.y[3]          0.00  0.17   -0.33   -0.11    0.00    0.11    0.33 1.00  2300
# nu.y[4]         -0.06  0.17   -0.40   -0.17   -0.06    0.06    0.29 1.00  2100
# nu.y[5]         -0.02  0.18   -0.38   -0.14   -0.02    0.10    0.34 1.00  5200
# nu.y[6]         -0.03  0.17   -0.37   -0.14   -0.03    0.08    0.30 1.00 20000
# nu.y[7]          0.01  0.18   -0.34   -0.10    0.01    0.13    0.36 1.00  5200
# nu.y[8]          0.08  0.18   -0.28   -0.04    0.08    0.20    0.45 1.00  6000
#
# SOME INTERPRETATION
# - intercepts nu close to zero as expected since data is centered at t=1
#-------------------------------------------------------------------------------

# Check for range of standardized factor loadings
range(est1[paste0("lambda.y.strd[", 1:12, "]"), 1])
# output: 0.5507665 0.8600209
# SOME INTERPRETATION: moderate to high factor loadings

# Check for the minimal effective sample size (validate convergence)
min(est1[est1[, "n.eff"] > 1, "n.eff"])
# output: 510 -> high enough for stable estimates
# SOME INTERPRETATION: good convergence

# Check for smallest Rhat statistic (validate convergence)
max(est1[est1[, "n.eff"] > 1, "Rhat"])
# output: 1.00586
# SOME INTERPRETATION: below threshold of 1.1 -> good convergence

# Calculate and Print covariance matrix of the latent factors
round(matrix(est1[c(
  paste0("rho.eta[", 1:4, ",1]"),
  paste0("rho.eta[", 1:4, ",2]"),
  paste0("rho.eta[", 1:4, ",3]"),
  paste0("rho.eta[", 1:4, ",4]")
), "mean"], 4, 4), 3)
#-------------------------------------------------------------------------------
#        [,1]   [,2]   [,3]   [,4]
# [1,]  1.000 -0.039 -0.154 -0.209
# [2,] -0.039  1.000  0.643  0.598
# [3,] -0.154  0.643  1.000  0.488
# [4,] -0.209  0.598  0.488  1.000
# SOME INTERPRETATION
# -> moderate to strong correlation between factors 2, 3 & 4
# -> small negative correlations between factor 1 and the other factors
#-------------------------------------------------------------------------------


#################### Parameter estimates at time point t=15 ####################

# Run jags model with burn-in parameter set to 5000
model2 <- jags.parallel(
  data = data_jags15, # data to use (t=15)
  parameters.to.save = params, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("01_cfa/models", "cfa_model1_4factors.txt")
)
# file containaining the model
# Store parameter estimates
est2 <- model2$BUGSoutput$summary
# Print parameter estimates
round(est2, 2)
#-------------------------------------------------------------------------------
# Inference for Bugs model "cfa_model1_4factors.txt", fit using jags,
# 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 5000 iterations saved:
#
# The output contains the mean of the parameters, their standard deviation (sd),
# several quantiles of their posterior distribution in %,
# the Gelman-Rubin statistic / potential scale reduction factor Rhat
# and a crude measure of effective sample size n.eff.
#                       mean  sd     2.5%   25%    50%    75%   97.5% Rhat n.eff
# lambda.y[1]           0.81  0.23   0.40   0.66   0.80   0.96   1.30 1.00   920
# lambda.y[2]           1.29  0.35   0.66   1.05   1.27   1.51   1.98 1.01   350
# lambda.y[3]           1.10  0.12   0.87   1.02   1.10   1.18   1.36 1.00  7700
# lambda.y[4]           0.79  0.13   0.54   0.70   0.78   0.88   1.07 1.00 17000
# lambda.y[5]           0.92  0.17   0.61   0.80   0.91   1.03   1.31 1.00  2100
# lambda.y[6]           0.86  0.18   0.53   0.74   0.85   0.97   1.24 1.00  2300
# lambda.y[7]           0.59  0.12   0.37   0.51   0.59   0.67   0.85 1.00  7000
# lambda.y[8]           0.71  0.16   0.42   0.61   0.71   0.81   1.04 1.00 20000
# lambda.y.strd[1]      0.71  0.08   0.54   0.65   0.71   0.76   0.84 1.00   680
# lambda.y.strd[2]      0.64  0.12   0.36   0.57   0.66   0.73   0.83 1.00  2100
# lambda.y.strd[3]      0.69  0.12   0.40   0.62   0.70   0.78   0.87 1.00   970
# lambda.y.strd[4]      0.90  0.04   0.81   0.88   0.90   0.92   0.95 1.00 20000
# lambda.y.strd[5]      0.91  0.03   0.84   0.90   0.92   0.94   0.96 1.00 20000
# lambda.y.strd[6]      0.77  0.08   0.59   0.73   0.78   0.82   0.89 1.00  8900
# lambda.y.strd[7]      0.80  0.07   0.65   0.76   0.81   0.85   0.91 1.00  7000
# lambda.y.strd[8]      0.79  0.08   0.60   0.75   0.80   0.84   0.91 1.00  3300
# lambda.y.strd[9]      0.75  0.09   0.55   0.70   0.76   0.82   0.89 1.00  2400
# lambda.y.strd[10]     0.88  0.05   0.76   0.85   0.88   0.91   0.95 1.00 13000
# lambda.y.strd[11]     0.70  0.10   0.48   0.65   0.72   0.77   0.86 1.00 11000
# lambda.y.strd[12]     0.68  0.11   0.43   0.62   0.69   0.76   0.85 1.00  4500
# rho.eta[1,1]          1.00  0.00   1.00   1.00   1.00   1.00   1.00 1.00     1
# rho.eta[2,1]         -0.19  0.20  -0.55  -0.34  -0.20  -0.06   0.21 1.00 11000
# rho.eta[3,1]         -0.22  0.20  -0.58  -0.36  -0.23  -0.09   0.18 1.00  2700
# rho.eta[4,1]         -0.11  0.21  -0.50  -0.26  -0.12   0.03   0.30 1.00 20000
# rho.eta[1,2]         -0.19  0.20  -0.55  -0.34  -0.20  -0.06   0.21 1.00 11000
# rho.eta[2,2]          1.00  0.00   1.00   1.00   1.00   1.00   1.00 1.00     1
# rho.eta[3,2]          0.82  0.07   0.66   0.79   0.83   0.87   0.92 1.00 14000
# rho.eta[4,2]          0.83  0.07   0.67   0.79   0.84   0.88   0.93 1.00 11000
# rho.eta[1,3]         -0.22  0.20  -0.58  -0.36  -0.23  -0.09   0.18 1.00  2700
# rho.eta[2,3]          0.82  0.07   0.66   0.79   0.83   0.87   0.92 1.00 14000
# rho.eta[3,3]          1.00  0.00   1.00   1.00   1.00   1.00   1.00 1.00     1
# rho.eta[4,3]          0.78  0.09   0.56   0.73   0.79   0.84   0.91 1.00  9200
# rho.eta[1,4]         -0.11  0.21  -0.50  -0.26  -0.12   0.03   0.30 1.00 20000
# rho.eta[2,4]          0.83  0.07   0.67   0.79   0.84   0.88   0.93 1.00 11000
# rho.eta[3,4]          0.78  0.09   0.56   0.73   0.79   0.84   0.91 1.00  9200
# rho.eta[4,4]          1.00  0.00   1.00   1.00   1.00   1.00   1.00 1.00     1
# sigma.eps[1]          0.18  0.06   0.11   0.15   0.18   0.21   0.31 1.00  2800
# sigma.eps[2]          0.16  0.05   0.09   0.13   0.15   0.18   0.26 1.00 19000
# sigma.eps[3]          0.31  0.10   0.16   0.24   0.30   0.37   0.54 1.00  4000
# sigma.eps[4]          0.13  0.04   0.08   0.11   0.13   0.16   0.23 1.00 20000
# sigma.eps[5]          0.13  0.04   0.07   0.10   0.12   0.15   0.22 1.00 20000
# sigma.eps[6]          0.23  0.07   0.14   0.19   0.22   0.27   0.39 1.00  6600
# sigma.eps[7]          0.29  0.10   0.16   0.23   0.28   0.34   0.51 1.00  7800
# sigma.eps[8]          0.26  0.08   0.14   0.21   0.25   0.31   0.45 1.00 11000
# sigma.eps[9]          0.28  0.08   0.16   0.23   0.27   0.33   0.48 1.00  3600
# sigma.eps[10]         0.22  0.07   0.11   0.16   0.20   0.25   0.39 1.00  8700
# sigma.eps[11]         0.25  0.07   0.14   0.20   0.24   0.29   0.42 1.00 11000
# sigma.eps[12]         0.41  0.12   0.23   0.33   0.39   0.47   0.69 1.00  4000
# sigma.eta[1,1]        0.19  0.07   0.09   0.14   0.18   0.23   0.35 1.00   740
# sigma.eta[2,1]       -0.07  0.07  -0.22  -0.11  -0.06  -0.02   0.07 1.00 13000
# sigma.eta[3,1]       -0.07  0.07  -0.23  -0.11  -0.07  -0.03   0.06 1.00  1900
# sigma.eta[4,1]       -0.04  0.08  -0.22  -0.09  -0.04   0.01   0.12 1.00 20000
# sigma.eta[1,2]       -0.07  0.07  -0.22  -0.11  -0.06  -0.02   0.07 1.00 13000
# sigma.eta[2,2]        0.59  0.17   0.32   0.46   0.56   0.68   1.00 1.00 20000
# sigma.eta[3,2]        0.47  0.15   0.24   0.36   0.45   0.55   0.83 1.00 20000
# sigma.eta[4,2]        0.55  0.17   0.29   0.43   0.53   0.64   0.96 1.00 17000
# sigma.eta[1,3]       -0.07  0.07  -0.23  -0.11  -0.07  -0.03   0.06 1.00  1900
# sigma.eta[2,3]        0.47  0.15   0.24   0.36   0.45   0.55   0.83 1.00 20000
# sigma.eta[3,3]        0.56  0.19   0.28   0.43   0.53   0.66   1.01 1.00  8900
# sigma.eta[4,3]        0.50  0.17   0.24   0.38   0.48   0.60   0.90 1.00 20000
# sigma.eta[1,4]       -0.04  0.08  -0.22  -0.09  -0.04   0.01   0.12 1.00 20000
# sigma.eta[2,4]        0.55  0.17   0.29   0.43   0.53   0.64   0.96 1.00 17000
# sigma.eta[3,4]        0.50  0.17   0.24   0.38   0.48   0.60   0.90 1.00 20000
# sigma.eta[4,4]        0.75  0.24   0.40   0.59   0.72   0.88   1.30 1.00  8500
# nu.y[1]               0.04  0.23  -0.37  -0.12   0.02   0.18   0.53 1.00   700
# nu.y[2]              -0.09  0.35  -0.71  -0.32  -0.11   0.13   0.61 1.01   320
# nu.y[3]              -0.17  0.12  -0.41  -0.24  -0.16  -0.09   0.05 1.00  6200
# nu.y[4]               0.14  0.12  -0.11   0.06   0.14   0.22   0.37 1.00 20000
# nu.y[5]              -0.56  0.24  -1.08  -0.71  -0.55  -0.40  -0.13 1.00  1100
# nu.y[6]              -0.58  0.25  -1.10  -0.73  -0.56  -0.41  -0.13 1.00  6200
# nu.y[7]               0.69  0.11   0.47   0.62   0.69   0.77   0.91 1.00  2300
# nu.y[8]               0.71  0.14   0.42   0.61   0.71   0.80   0.98 1.00 16000
#-------------------------------------------------------------------------------

# Check for range of standardized factor loadings
range(est2[paste0("lambda.y.strd[", 1:12, "]"), 1])
# output: 0.6427082 0.9145236
# SOME INTERPRETATION: moderate to high factor loadings

# Check for the minimal effective sample size (validate convergence)
min(est2[est2[, "n.eff"] > 1, "n.eff"])
# output: 320 -> high enough for stable estimates
# SOME INTERPRETATION: good convergence

# Check for smallest Rhat statistic (validate convergence)
max(est2[est2[, "n.eff"] > 1, "Rhat"])
# output:  1.01068 -> below threshold of 1.1
# SOME INTERPRETATION: good convergence

# Calculate and Print covariance matrix of the latent factors
round(matrix(est2[c(
  paste0("rho.eta[", 1:4, ",1]"),
  paste0("rho.eta[", 1:4, ",2]"),
  paste0("rho.eta[", 1:4, ",3]"),
  paste0("rho.eta[", 1:4, ",4]")
), "mean"], 4, 4), 3)
#-------------------------------------------------------------------------------
#        [,1]   [,2]   [,3]   [,4]
# [1,]  1.000 -0.195 -0.223 -0.112
# [2,] -0.195  1.000  0.823  0.829
# [3,] -0.223  0.823  1.000  0.776
# [4,] -0.112  0.829  0.776  1.000
# SOME INTERPRETATION
# -> very strong correlations between factors 2, 3 and 4
# -> small negative correlations between factor 4
#    and the other factors
#-------------------------------------------------------------------------------


################################################################################
# CONVERGENCE Checks
################################################################################

####################### Convergence at time point t=1 ##########################

# Run jags model at t=1 with burn-in parameter set to 1
model1_conv <- jags.parallel(
  data = data_jags1, # data to use (t=1)
  parameters.to.save = params, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file = here::here("01_cfa/models", "cfa_model1_4factors.txt")
)

# Create Markov Chain Monum_timee Carlo samples
samps1 <- as.mcmc(model1_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("01_cfa/convergence_checks",
               "cfa_model1_gelmanplot.pdf"))
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
pdf(here::here("01_cfa/convergence_checks",
               "cfa_model1_xyplot.pdf"), pointsize = 6)
plot(samps1)
# When the model is converged after the burn-in phase,
# the trace should not exhibit any increasing or decreasing trends,
# the differenum_time chains (plotted in differenum_time colors) should mix well
# and the density plot should be smooth and have only a single peak.
dev.off() # reset layout

# -> The model seems to be converged after 5000 MCMC iterations.

####################### Convergence at time point t=10 #########################

# Run jags model at t=10 with burn-in parameter set to 1
model2_conv <- jags.parallel(
  data = data_jags15, # data to use (t=10)
  parameters.to.save = params, # parameters to output
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file = here::here("01_cfa/models", "cfa_model1_4factors.txt")
)

# Create Markov Chain Monum_timee Carlo samples
samps2 <- as.mcmc(model2_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("01_cfa/convergence_checks",
               "cfa_model2_gelmanplot.pdf"))
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
pdf(here::here("01_cfa/convergence_checks",
               "cfa_model2_xyplot.pdf"), pointsize = 6)
plot(samps2)
dev.off() # reset layout

# -> The model seems to be converged after 5000 MCMC iterations.
