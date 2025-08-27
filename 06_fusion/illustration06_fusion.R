# Library needed to run the jags code
library(R2jags)
# You might need to run install.packages('R2jags')
# and install.packages('rjags')

# Library to manage file paths
#setwd("c:/holger/sem/dsem_tutorial")
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
# - MODEL:   Latent state switching AR(1) 2-level model with person-specific factor structure
#             state 1 corresponding to three factors
#             state 2 corresponding to a single factor
# -> dlcsem_model1_3factor_ar1_2level-int.txt
#
# - PLOTS
#
# - CONVERGENCE checks
################################################################################


################################################################################
# DATA preparation
################################################################################
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
data_3factor <- list(
  N = num_obs, # sample size
  Nt = num_time, # time points
  y = y_data, # responses to all items, jags file uses 4:12 
  psi0 = diag(3), # hyperprior wishart distribution
  mu.zeta2 = rep(0, 3)  # hyperprior multivariate normal distribution
)


################################################################################
# MODEL:  One-factor state-switching AR(1) with person-specific random intercept
################################################################################

# Define parameters which are being estimated
params_model1 <- c(
  "alpha.S1", "alpha.S2", "beta.S1", "beta.S2",
  "lambda.y.S1", "lambda.y.S2", "nu.y.S1", "nu.y.S2", 
  "sigma.y", "sigma.zeta21.S1","sigma.zeta21.S2",
  "sigma.eta.S1", "sigma.eta.S2", "P2", "b2"
)
# alpha.S1: vector of first AR(1) parameter of state 1 (random intercept/baseline value)
# alpha.S2: first AR(1) parameter of state 2 (random intercept/baseline value)
# beta.S1: matrix of second AR(1) parameter in state 1 (slope/autoregressive and cross-lagged coefficients)
# beta.S2: second AR(1) parameter in state 2 (slope/autoregressive coefficient)
# lambda.y.S1: factor loadings in state 1
# lambda.y.S2: factor loadings in state 2
# nu.y.S1: factor intercepts in state 1
# nu.y.S2: factor intercepts in state 1
# sigma.y: variance of the residuals
# sigma.zeta21.S1: covariance matrix of the random intercepts in state 1
# sigma.zeta21.S2: variance of the random intercept in state 2
# sigma.eta.S1: covariance matrix of the latent factors in state 1
# sigma.eta.S2: variance of the latent factor in state 2
# P2: transition probabilities from state 2
# b2: parameter for the transition probabilities from state 2


###############################################################
# Model 1: one factor, test treatment effect heterogeneity with BAI factor
# convergence check
###############################################################
# Run jags model for items 4-12 (all WAI indicators)
model1 <- jags.parallel(
  data = data_3factor, # data to use
  parameters.to.save = params_model1, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("06_fusion/models",
                          "dlcsem_model1_3factor_ar1_2level-int.txt")
)

# save summary results
#saveRDS(model1$BUGSoutput$summary,"06_fusion/results/results_illustration06.RDS")


# Store parameter estimates
est1 <- model1$BUGSoutput$summary
#or if reading from the results folder
est1 <- readRDS("06_fusion/results/results_illustration06.RDS")


# Print parameter estimates
round(est1, 2)

# Inference for Bugs model at "C:/holger/SEM/dsem_tutorial/06_fusion/models/dlcsem_model1_3factor_ar1_2level-int.txt", fit using jags,
# 4 chains, each with 10000 iterations (first 5000 discarded)
# n.sims = 20000 iterations saved. Running time = secs
# mean    sd    2.5%     25%     50%     75%   97.5% Rhat n.eff
# P2[1]                   0.02  0.01    0.00    0.01    0.02    0.03    0.05 1.08    36
# P2[2]                   0.98  0.01    0.95    0.97    0.98    0.99    1.00 1.10    34
# alpha.S1[1]             0.22  0.08    0.08    0.17    0.22    0.27    0.38 1.27    14
# alpha.S1[2]             0.33  0.08    0.20    0.27    0.32    0.38    0.50 1.08    42
# alpha.S1[3]             0.38  0.08    0.21    0.32    0.39    0.44    0.54 1.20    20
# alpha.S2                0.53  0.12    0.29    0.45    0.53    0.61    0.77 1.04    71
# b2[1]                   2.32  0.21    1.91    2.17    2.31    2.45    2.76 1.04    68
# b2[2]                   0.04  0.77   -1.48   -0.47    0.05    0.56    1.54 1.00   710
# b2[3]                  -0.20  0.80   -1.82   -0.73   -0.18    0.34    1.33 1.03    85
# b2[4]                  -0.37  0.82   -2.01   -0.93   -0.36    0.19    1.19 1.02   160
# beta.S1[1,1]            0.20  0.17   -0.13    0.09    0.20    0.31    0.52 1.02   160
# beta.S1[2,1]            0.07  0.13   -0.19   -0.02    0.07    0.16    0.32 1.01   220
# beta.S1[3,1]            0.09  0.16   -0.23   -0.02    0.08    0.19    0.42 1.00 15000
# beta.S1[1,2]            0.05  0.17   -0.28   -0.06    0.05    0.16    0.38 1.02   150
# beta.S1[2,2]            0.16  0.17   -0.16    0.05    0.16    0.27    0.52 1.01   620
# beta.S1[3,2]            0.20  0.17   -0.14    0.08    0.20    0.31    0.55 1.03    81
# beta.S1[1,3]            0.12  0.14   -0.16    0.03    0.12    0.22    0.39 1.02   200
# beta.S1[2,3]            0.21  0.13   -0.04    0.12    0.21    0.29    0.49 1.01   240
# beta.S1[3,3]            0.11  0.16   -0.22    0.00    0.11    0.22    0.42 1.02   270
# beta.S2                 0.61  0.12    0.38    0.53    0.61    0.69    0.85 1.01   470
# deviance             9667.82 79.52 9509.42 9614.84 9668.25 9722.43 9821.19 1.02   150
# lambda.y.S1[1]          1.23  0.08    1.09    1.18    1.23    1.28    1.39 1.02   160
# lambda.y.S1[2]          1.21  0.08    1.07    1.16    1.21    1.27    1.37 1.01   190
# lambda.y.S1[3]          1.06  0.08    0.92    1.01    1.06    1.12    1.23 1.01   200
# lambda.y.S1[4]          1.03  0.09    0.84    0.96    1.03    1.09    1.21 1.09    34
# lambda.y.S1[5]          1.48  0.15    1.24    1.37    1.46    1.57    1.80 1.17    22
# lambda.y.S1[6]          0.75  0.12    0.55    0.67    0.74    0.82    1.00 1.12    27
# lambda.y.S2[1]          1.06  0.06    0.95    1.02    1.06    1.10    1.18 1.00  1400
# lambda.y.S2[2]          0.98  0.06    0.87    0.94    0.98    1.02    1.10 1.02   160
# lambda.y.S2[3]          0.67  0.06    0.55    0.63    0.67    0.71    0.79 1.05    58
# lambda.y.S2[4]          0.68  0.05    0.58    0.64    0.68    0.72    0.79 1.09    32
# lambda.y.S2[5]          0.83  0.06    0.72    0.79    0.83    0.87    0.95 1.01   200
# lambda.y.S2[6]          1.37  0.07    1.23    1.32    1.37    1.42    1.52 1.01   340
# lambda.y.S2[7]          1.19  0.07    1.06    1.14    1.19    1.23    1.32 1.07    39
# lambda.y.S2[8]          1.03  0.07    0.89    0.98    1.03    1.07    1.17 1.01   280
# nu.y.S1[1]             -0.10  0.05   -0.20   -0.13   -0.10   -0.07   -0.01 1.02   320
# nu.y.S1[2]             -0.14  0.05   -0.24   -0.17   -0.14   -0.11   -0.05 1.02   160
# nu.y.S1[3]             -0.44  0.05   -0.53   -0.47   -0.43   -0.40   -0.35 1.01   220
# nu.y.S1[4]             -0.25  0.05   -0.35   -0.28   -0.25   -0.21   -0.14 1.04    76
# nu.y.S1[5]             -0.24  0.10   -0.46   -0.30   -0.23   -0.17   -0.07 1.16    23
# nu.y.S1[6]              0.03  0.08   -0.13   -0.02    0.04    0.08    0.17 1.12    27
# nu.y.S2[1]             -0.19  0.06   -0.31   -0.23   -0.19   -0.15   -0.08 1.01   550
# nu.y.S2[2]             -0.06  0.06   -0.17   -0.09   -0.06   -0.02    0.05 1.01   660
# nu.y.S2[3]              0.76  0.05    0.65    0.73    0.76    0.80    0.87 1.02   140
# nu.y.S2[4]              0.15  0.05    0.06    0.12    0.15    0.18    0.24 1.07    40
# nu.y.S2[5]             -0.11  0.05   -0.21   -0.14   -0.11   -0.07   -0.01 1.01   620
# nu.y.S2[6]             -0.58  0.07   -0.72   -0.62   -0.58   -0.53   -0.45 1.00  1200
# nu.y.S2[7]             -0.10  0.06   -0.22   -0.14   -0.10   -0.06    0.02 1.06    51
# nu.y.S2[8]              0.35  0.06    0.22    0.31    0.35    0.39    0.47 1.00   900
# sigma.eta.S1[1,1]       0.15  0.02    0.11    0.13    0.15    0.16    0.19 1.02   130
# sigma.eta.S1[2,1]       0.10  0.02    0.07    0.09    0.10    0.11    0.13 1.02   120
# sigma.eta.S1[3,1]       0.08  0.01    0.05    0.07    0.08    0.09    0.11 1.04    80
# sigma.eta.S1[1,2]       0.10  0.02    0.07    0.09    0.10    0.11    0.13 1.02   120
# sigma.eta.S1[2,2]       0.17  0.03    0.12    0.15    0.17    0.19    0.24 1.05    56
# sigma.eta.S1[3,2]       0.07  0.01    0.05    0.06    0.07    0.08    0.10 1.02   210
# sigma.eta.S1[1,3]       0.08  0.01    0.05    0.07    0.08    0.09    0.11 1.04    80
# sigma.eta.S1[2,3]       0.07  0.01    0.05    0.06    0.07    0.08    0.10 1.02   210
# sigma.eta.S1[3,3]       0.11  0.02    0.08    0.10    0.11    0.12    0.15 1.08    44
# sigma.eta.S2            0.06  0.01    0.04    0.05    0.05    0.06    0.07 1.00  5100
# sigma.y[1]              0.27  0.02    0.24    0.26    0.27    0.28    0.31 1.00  1500
# sigma.y[2]              0.19  0.02    0.17    0.18    0.19    0.20    0.23 1.02   130
# sigma.y[3]              0.23  0.02    0.20    0.22    0.23    0.24    0.27 1.00   800
# sigma.y[4]              0.32  0.02    0.28    0.31    0.32    0.34    0.37 1.02   150
# sigma.y[5]              0.21  0.02    0.18    0.20    0.21    0.22    0.24 1.03    99
# sigma.y[6]              0.27  0.02    0.23    0.26    0.27    0.29    0.31 1.04    65
# sigma.y[7]              0.35  0.02    0.31    0.34    0.35    0.37    0.40 1.02   150
# sigma.y[8]              0.19  0.02    0.15    0.17    0.18    0.20    0.23 1.01   210
# sigma.y[9]              0.50  0.03    0.45    0.48    0.50    0.52    0.57 1.00   690
# sigma.zeta21.S1[1,1]    0.20  0.05    0.12    0.16    0.20    0.23    0.33 1.01   200
# sigma.zeta21.S1[2,1]    0.10  0.04    0.03    0.07    0.09    0.12    0.20 1.01   490
# sigma.zeta21.S1[3,1]    0.10  0.05    0.02    0.07    0.09    0.13    0.22 1.06    50
# sigma.zeta21.S1[1,2]    0.10  0.04    0.03    0.07    0.09    0.12    0.20 1.01   490
# sigma.zeta21.S1[2,2]    0.21  0.06    0.12    0.17    0.20    0.25    0.37 1.02   130
# sigma.zeta21.S1[3,2]    0.06  0.05   -0.02    0.03    0.06    0.09    0.16 1.03   100
# sigma.zeta21.S1[1,3]    0.10  0.05    0.02    0.07    0.09    0.13    0.22 1.06    50
# sigma.zeta21.S1[2,3]    0.06  0.05   -0.02    0.03    0.06    0.09    0.16 1.03   100
# sigma.zeta21.S1[3,3]    0.24  0.08    0.12    0.18    0.23    0.29    0.42 1.13    23
# sigma.zeta21.S2         0.46  0.13    0.27    0.37    0.45    0.53    0.75 1.04    68
#
# SOME INTERPRETATION
# - P2[1]: Low chance of reverting from S2 to S1
# - b2: No predictive elements in HMM
# - beta.S1: Low  temporal stability in S1
# - beta.S2: High  temporal stability in S2
# - lambda.y.S1/.S2: Similar size for factor loadings across S and items
#-------------------------------------------------------------------------------

# Check for the minimal effective sample size (validate convergence)
min(est1[est1[, "n.eff"] > 1, "n.eff"])
median(est1[est1[, "n.eff"] > 1, "n.eff"])
# output: 14 -> rather small
# median: 150 is ok.

# Check for smallest Rhat statistic (validate convergence)
max(est1[est1[, "n.eff"] > 1, "Rhat"])
# output: 1.268372 (one intercept)
#sum(est1[est1[, "n.eff"] > 1, "Rhat"]>1.1)
sum(est1[est1[, "n.eff"] > 1, "Rhat"]>1.2)
# 1 value above 1.2 (lenient criterion)
# -> increasing the iterations would improve this (but takes time)
#-------------------------------------------------------------------------------

# # Define the factor structure for standardization
# factor_structure_state1 <- data.frame(
#   indicator = 1:9,
#   factor = c(rep(1, 3),rep(2, 3),rep(3, 3)),
#   loading_id = c(NA, 1, 2,NA,3,4,NA,5,6)  # first is fixed
# )
# 
# factor_structure_state2 <- data.frame(
#   indicator = 1:9,
#   factor = rep(1, 9),
#   loading_id = c(NA, 1, 2,3,4,5,6,7,8,9)  # first is fixed
# )

# # Calculate standardized factor loadings for state 1
# # (see code in utils/standardization.R)
# result_state1 <- run_factor_analysis_map(
#   est = est1,
#   factor_structure = factor_structure_state1,
#   n_factors = 3,
#   beta_prefix = "beta.S1",
#   sigma_eta_prefix = "sigma.eta.S1",
#   sigma_y_prefix = "sigma.y",
#   lambda_prefix = "lambda.y.S1",
#   person_intercept_var_name = "sigma.zeta21"
# )
# print_analysis_results(result_state1)
# 
# # Calculate standardized factor loadings for state 2
# result_state2 <- run_factor_analysis_map(
#   est = est1,
#   factor_structure = factor_structure_state2,
#   n_factors = 1,
#   beta_prefix = "beta.S2",
#   sigma_eta_prefix = "sigma.eta.S2",
#   sigma_y_prefix = "sigma.y",
#   lambda_prefix = "lambda.y.S2",
#   person_intercept_var_name = "sigma.zeta21"
# )
# print_analysis_results(result_state2)


#-------------------------------------------------------------------------------
# Variances, covariances, correlations and ICC
#-------------------------------------------------------------------------------
# Residual covariance matrix L1 in state 1
sigma.eta.S1 <- matrix(est1[c(paste0("sigma.eta.S1[",1:3,",1]"),
                            paste0("sigma.eta.S1[",1:3,",2]"),
                            paste0("sigma.eta.S1[",1:3,",3]")),"mean"],3,3)
# Random intercepts covariance matrix  L2 in state 1
sigma.zeta21.S1 <- matrix(est1[c(paste0("sigma.zeta21.S1[",1:3,",1]"),
                                 paste0("sigma.zeta21.S1[",1:3,",2]"),
                                 paste0("sigma.zeta21.S1[",1:3,",3]")),"mean"],3,3)

# Residual variance L1 in state 2
sigma.eta.S2 <- matrix(est1[c(paste0("sigma.eta.S2")),"mean"],1,1)
# Random intercept variance L2 in state 2
sigma.zeta21.S2 <- matrix(est1[c(paste0("sigma.zeta21.S2")),"mean"],1,1)

############################
# State 1
############################
# Define transition matrix beta.S1
beta <-matrix(est1[c(paste0("beta.S1[",1:3,",1]"),paste0("beta.S1[",1:3,",2]"),
                     paste0("beta.S1[",1:3,",3]")),1],3,3,byrow=F)



# Factor covariance matrix L1
sigma.eta.S1.total <- get.cov.from.AR1(beta, sigma.eta.S1)

# Factor covariance matrix L1 + L2 (for standardization)
Sigma_total.S1 <- sigma.eta.S1.total+sigma.zeta21.S1

# print covariance matrices L1 and L2 separately and combined
round(sigma.eta.S1.total,3) #L1
round(sigma.zeta21.S1,3)    #L2
round(Sigma_total.S1,3)     #L1+L2

# correlation matrix L1 and L2 separately and combined
round(corfun(sigma.eta.S1.total),3) #L1
round(corfun(sigma.zeta21.S1),3)    #L2
round(corfun(Sigma_total.S1),3)     #L1+L2

# icc 
diag(round(sigma.zeta21.S1/(Sigma_total.S1),3))

############################
# State 2
############################
# Factor covariance matrix L1
sigma.eta.S2.total <- sigma.eta.S2/(1-est1[paste0("beta.S2"),1]^2)

# Factor variance L1 + L2 (for standardization)
Sigma_total.S2 <- sigma.eta.S2.total+sigma.zeta21.S2

# print variances L1 and L2 separately and combined
round(sigma.eta.S2.total,3) #L1
round(sigma.zeta21.S2,3)    #L2
round(Sigma_total.S2,3)     #L1+L2

# icc
round(sigma.zeta21.S2/(Sigma_total.S2),3)



#-------------------------------------------------------------------------------
# standardized factor loadings (using the total variances from above)
#-------------------------------------------------------------------------------
lambda.y.S1.std <- lambda.y.S2.std <- c()

############################
# extract parameters
############################
# complete variances for standardization
vxi1 <- diag(Sigma_total.S1) #state 1
vxi2 <- diag(Sigma_total.S2) #state 2
# residual variances
vx1 <- vx2 <- est1[paste0("sigma.y[",1:9,"]"),1]
# extract factor loadings and add the 1 for scaling indicators
lambda.y.S1 <- c(1,est1[paste0("lambda.y.S1[",1:2,"]"),1],
                 1,est1[paste0("lambda.y.S1[",3:4,"]"),1],
                 1,est1[paste0("lambda.y.S1[",5:6,"]"),1])
lambda.y.S2 <- c(1,est1[paste0("lambda.y.S2[",1:8,"]"),1])

############################
# standardization state 1
############################
lambda.y.S1.std[1:3+0] <- lambda.y.S1[1:3+0]*sqrt(vxi1[1]/(lambda.y.S1[1:3+0]^2*vxi1[1]+vx1[1:3+0]))
lambda.y.S1.std[1:3+3] <- lambda.y.S1[1:3+3]*sqrt(vxi1[2]/(lambda.y.S1[1:3+3]^2*vxi1[2]+vx1[1:3+3]))
lambda.y.S1.std[1:3+6] <- lambda.y.S1[1:3+6]*sqrt(vxi1[3]/(lambda.y.S1[1:3+6]^2*vxi1[3]+vx1[1:3+6]))

# print results
round(lambda.y.S1.std,3)

############################
# standardization state 2
############################
lambda.y.S2.std[1:9] <- lambda.y.S2[1:9]*sqrt(vxi2[1]/(lambda.y.S2[1:9]^2*vxi2[1]+vx2[1:9]))

# print results
round(lambda.y.S2.std,2)

# see how they differ across states
round(cbind(lambda.y.S1.std,lambda.y.S2.std),2)

# get a range
range(lambda.y.S1.std)
range(lambda.y.S2.std)

#-------------------------------------------------------------------------------
# regression coefficents
#-------------------------------------------------------------------------------

# state 1
# Matrix, notation: beta[j,k] indicates that factor j at (t-1) affects factor k at (t) 
round(matrix(est1[c(paste0("beta.S1[",1:3,",1]"),
                    paste0("beta.S1[",1:3,",2]"),
                    paste0("beta.S1[",1:3,",3]")),1],3,3,byrow=F),3)
# INTERPRETATION: all very small

# check credible intervals
round(est1[c(paste0("beta.S1[",1:3,",1]"),
             paste0("beta.S1[",1:3,",2]"),
             paste0("beta.S1[",1:3,",3]")),c("2.5%","97.5%")],3)
# INTERPRETATION: all CIs cover zero

# state 2
round(est1[paste0("beta.S2"),],3)
# way larger and CI is (0.38,0.85)

#-------------------------------------------------------------------------------
# HMM
#-------------------------------------------------------------------------------
round(est1[paste0("b2[",1:4,"]"),],2)
# INTERPRETATION: CIs cover zero for all predictors (b2[2:4])

# odds ratios (exp(b2))
round(exp(est1[paste0("b2[",1:4,"]"),1]),2)


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
  data = data_3factor, # data to use
  parameters.to.save = params_model1_plots, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("06_fusion/models",
                          "dlcsem_model1_3factor_ar1_2level-int.txt")
)


fscores_illu6 <- model1_plots$BUGSoutput$summary
# save summary results for factor scores
#saveRDS(fscores_illu6,"06_fusion/results/results_illustration06_factorscores.RDS")

# if not run, read in the results
fscores_illu6 <- readRDS("06_fusion/results/results_illustration06_factorscores.RDS")


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
s_hat <- matrix(fscores_illu6[names_s, "50%"],
                nrow = num_obs, ncol = num_time, byrow = FALSE)
ps_hat <- matrix(fscores_illu6[names_ps, "mean"],
                 nrow = num_obs, ncol = num_time - 1, byrow = FALSE)
ps_hat <- cbind(1, ps_hat)
#eta_s1 <- matrix(fscores_illu6[names_eta_s1, "mean"],
#                 nrow = num_obs, ncol = num_time, byrow = TRUE)
#eta_s2 <- matrix(fscores_illu6[names_eta_s2, "mean"],
#                 nrow = num_obs, ncol = num_time, byrow = TRUE)
# We did not include a plot for the factor scores

########################## State-swiching over time ############################

pdf(
  file = here::here("06_fusion/plots",
                    "stateswitch_Fusion.pdf"),
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





################################################################################
# COMPARISON TO ILLUSTRATION 5
################################################################################

# switch_points for illustration 6 (Fusion)
switch_points_illu6 <- rep(NA, num_obs)
for (i in 1:num_obs) {
  for (j in 2:num_time) {
    if (s_hat[i, j] > s_hat[i, j - 1]) {  # Detect a switch from state 1 to 2
      switch_points_illu6[i] <- j
      break  # Stop at the first switch point
    }
  }
}

# get factorscores from illustration 5
fscores_illu5 <- readRDS("05_dlcsem/results/results_illustration05_factorscores.RDS")

# extract state membership
s_hat_illu5 <- matrix(fscores_illu5[names_s,"50%"],N,Nt,byrow=F)

# switch_points for illustration 5 (dlcsem)
switch_points_illu5 <- rep(NA, num_obs)
for (i in 1:num_obs) {
  for (j in 2:num_time) {
    if (s_hat_illu5[i, j] > s_hat_illu5[i, j - 1]) {  # Detect a switch from state 1 to 2
      switch_points_illu5[i] <- j
      break  # Stop at the first switch point
    }
  }
}

# average time point for the first switch (if switch occurs)
mean(switch_points_illu6,na.rm=T)
mean(switch_points_illu5,na.rm=T)

cbind(switch_points_illu5,switch_points_illu6)

# these are the ones that switch or not in illustration 6 (filter)
switch0 <- switch
switch0[is.na(switch)] <- 0
switch0[switch0>0] <- 1

# scatter plot (not in article)
scatter.smooth(jitter(switch_points_illu5),jitter(switch_points_illu6),xlab="Switch Illu 5",ylab="Switch Illu 6",
               xlim=c(1,15),ylim=c(1,15))
abline(0,1,lty=2)
abline(lm(switch_points_illu6~switch_points_illu5),lty=3)
legend("topright",c("loess","diagonal","regression"),lty=1:3)

cor(switch_points_illu5,switch_points_illu6,use="pair")
# correlation is close to zero




####################################################################
# APPENDIX opposite DE-FUSION
# this model reverses states (so persons start with global factor 
# and switch to 3-factors)
####################################################################
model1_opposite <- jags.parallel(
  data = data_3factor, # data to use
  parameters.to.save = params_model1, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("06_fusion/models/appendix",
                          "dlcsem_model1_3factor_ar1_2level-int_opposite.txt")
)

#saveRDS(model1_opposite$BUGSoutput$summary,"06_fusion/results/results_illustration06_opposite.RDS")
# or read the results
model1_opposite <- readRDS("06_fusion/results/results_illustration06_opposite.RDS")

# paste results
model1_opposite

# Re-run jags model for items 10-12 (all BAI indicators)
model1_opposite_plots <- jags.parallel(
  data = data_3factor, # data to use
  parameters.to.save = params_model1_plots, # output parameters
  n.iter = 10000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip
  # for avoiding autocorrelation (in this case None)
  n.burnin = 5000, # burn-in parameter
  model.file = here::here("06_fusion/models/appendix",
                          "dlcsem_model1_3factor_ar1_2level-int_opposite.txt")
)

fscores_opposite <- model1_opposite_plots$BUGSoutput$summary

#saveRDS(fscores_opposite,"06_fusion/results/results_illustration06_factorscores_opposite.RDS")
# or read the results
fscores_opposite <- readRDS("06_fusion/results/results_illustration06_factorscores_opposite.RDS")


# extract state membership
s_hat_opposite <- matrix(fscores_opposite[names_s,"50%"],N,Nt,byrow=F)

# recode so that 
# state 1 refers to three factor model
# state 2 refers to single factor model
# which allows us to better compare it to the original model
s_hat_opposite.re <- 3 - s_hat_opposite



####################################################################
# individual state memberships
####################################################################
pdf(file = here::here("06_fusion/plots",
                      "States_fusion_opposite_all.pdf"),width=2*3,height=2*4)

par(mfrow=c(4,3))
for(i in 1:N){
  plot(jitter(s_hat_opposite.re[i,],factor=.2),type="l",ylim=c(.9,2.1),axes=F,ylab="",xlab="",col="green",lty=1,main=i)
  par(new=T)
  plot(jitter(s_hat[i,],factor=.2),type="l",ylim=c(.9,2.1),axes=F,ylab="",xlab="",col="black",lty=1)
  axis(1)
  axis(2,c(1,2),paste0(c(1,2)))
  legend("top",c("original","opposite"),lty=1,col=c("black","green"))
}

dev.off()

pdf(file = here::here("06_fusion/plots",
                      "States_fusion_opposite_selected.pdf"),width=2*3,height=2*2)

par(mfrow=c(2,3))
for(i in c(3,4,9,19,35,42)){
  plot(jitter(s_hat_opposite.re[i,],factor=.2),type="l",ylim=c(.9,2.1),axes=F,ylab=expression(hat(S)),xlab="Session",col="green",lty=1,main=i)
  par(new=T)
  plot(jitter(s_hat[i,],factor=.2),type="l",ylim=c(.9,2.1),axes=F,ylab="",xlab="",col="black",lty=1)
  axis(1)
  axis(2,c(1,2),paste0(c(1,2)))
  if(i ==3){
    legend("top",c("original","opposite"),lty=1,col=c("black","green"),bty="n")
  }
}

dev.off()

# this is an interesting plot.
# the data overrides the model in many instances,
# persons go from S=1 to S=2 directly at t=2, then later switch back to S=1
# -> we force them to start in the wrong state. they choose similarly to what we modeled above


####################################################################
# sensitivity, specificity, and accuracy over time (assuming the fusion model is correct)
####################################################################
conf_matrix <- list()
sensitivity <- specificity <- accuracy <- c()
for(i in 2:Nt){#i<-2
  # Confusion matrix
  conf_matrix[[i]] <- table(Actual = s_hat[,i], Predicted = s_hat_opposite.re[,i])
  
  # true/false negative/positive for each time point
  TN <- conf_matrix[[i]][1, 1]
  FN <- conf_matrix[[i]][2, 1]
  FP <- conf_matrix[[i]][1, 2]
  TP <- conf_matrix[[i]][2, 2]
  
  # compute sensitivity, specificity, accuracy for each time point
  sensitivity[i] <- TP / (TP + FN)
  specificity[i] <- TN / (TN + FP)
  accuracy[i]    <- (TP + TN) / sum(conf_matrix[[i]])
}

round(sensitivity,2)
round(specificity,2)
round(accuracy,2)

pdf(file = here::here("06_fusion/plots",
                      "states_fusion_opposite_accuracy.pdf"),width=6,height=6)

par(mfrow=c(1,1))
plot(2:Nt,sensitivity[-1],ylim=c(0,1),ylab="",xlab="Session",axes=F,type="l",lty=2)
par(new=T)
plot(2:Nt,specificity[-1],ylim=c(0,1),ylab="",xlab="",axes=F,type="l",lty=3)
par(new=T)
plot(2:Nt,accuracy[-1],ylim=c(0,1),ylab="",xlab="",axes=F,type="l",lty=1)
axis(1);axis(2)
legend("bottomright",c("Sensitivity","Specificity","Accuracy"),lty=c(2,3,1),bty="n")

dev.off()


################################################################################
# CONVERGENCE checks
################################################################################

############################ Convergence Model 1 ###############################

# Run jags model 1 with burn-in parameter set to 1
model1_conv <- jags.parallel(
  data = data_3factor, # data to use (t=1)
  parameters.to.save = params_model1, # parameters to output
  n.iter = 20000, # number of iterations
  n.chains = 4, # number of chains
  n.thin = 1, # number of iterations to skip for
  # avoiding autocorrelation (in this case None)
  n.burnin = 1, # burn-in parameter
  model.file =  here::here("06_fusion/models",
                           "dlcsem_model1_3factor_ar1_2level-int.txt")
)


# Create Markov Chain Monte Carlo samples
samps1 <- as.mcmc(model1_conv)

# Create Gelman-Rubin diagnostic plots
pdf(here::here("06_fusion/convergence_checks",
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
pdf(file = here::here("06_fusion/convergence_checks",
                      "dlcsem_model1_xyplot.pdf"), pointsize = 6)
plot(samps1)
# When the model is converged after the burn-in phase,
# the trace should not exhibit any increasing or decreasing trends,
# the different chains (plotted in different colors) should mix well
# and the density plot should be smooth and have only a single peak.
dev.off() # reset layout

# -> The model seems to be converged after 10000 MCMC iterations. 
# A few parameters have Rhat 1.2 at that point, but it needs another 40k to reduce that
# (which will take several hours to run)






