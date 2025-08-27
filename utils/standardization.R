library(expm)

#--------------------------------------------------
# Wrapper function to run full analysis
#--------------------------------------------------
run_factor_analysis_map <- function(est,
                                    factor_structure,
                                    n_factors = 1,
                                    beta_prefix = "beta",
                                    sigma_eta_prefix = "sigma.eta",
                                    sigma_y_prefix = "sigma.y",
                                    lambda_prefix = "lambda.y",
                                    person_intercept_var_name = NULL,
                                    person_slope_var_name = NULL,
                                    time_slope_var_name = NULL) {
  # Step 1: Compute latent variance
  latent_var_out <- compute_latent_var_dynamic(
    est = est,
    beta_prefix = beta_prefix,
    sigma_eta_prefix = sigma_eta_prefix,
    n_factors = n_factors,
    person_intercept_var_name = person_intercept_var_name,
    person_slope_var_name = person_slope_var_name,
    time_slope_var_name = time_slope_var_name
  )
  total_latent_var <- if (n_factors == 1) {
    latent_var_out$tot_var
  } else {
    diag(latent_var_out$tot_var)
  }

  # Step 3: Compute standardized loadings
  std_loadings <- compute_strd_loadings(
    est = est,
    factor_structure = factor_structure,
    var_latent_diag = total_latent_var,
    sigma_y_prefix = sigma_y_prefix,
    lambda_prefix = lambda_prefix
  )

  # Step 4: Compute ICC (optional)
  residual_vars <- sapply(factor_structure$indicator, function(j) {
    est[paste0(sigma_y_prefix, "[", j, "]"), "mean"]
  })
  icc <- compute_icc(total_latent_var, mean(residual_vars))

  return(list(
    std_loadings = std_loadings,
    icc = icc,
    total_latent_var = total_latent_var,
    breakdown = latent_var_out
  ))
}

# Print the results
print_analysis_results <- function(result) {
  cat("--------------------------------------------------\n")
  cat("Standardized Factor Loadings:\n")
  print(format(round(result$std_loadings, 3), justify = "right"))
  cat("\n")
  cat("Intraclass Correlation Coefficient (ICC):\n")
  cat(sprintf("%.3f\n\n", result$icc))
  cat("Total Latent Variance (Factor Level):\n")
  cat(sprintf("  %.3f\n", result$total_latent_var))
  cat("  Splitted into:\n")
  cat("    Raw Dynamic (ARMA) Factor Variance:\n")
  cat(sprintf("      %.3f\n", result$breakdown$dynamic_var))
  if (!is.null(result$breakdown$person_intercept_var)) {
    cat("    Raw Person-Level Intercept Variance:\n")
    cat(sprintf("      %.3f\n", result$breakdown$person_intercept_var))
  }
  if (!is.null(result$breakdown$person_slope_var)) {
    cat("    Raw Person-Level Slope Variance:\n")
    cat(sprintf("      %.3f\n", result$breakdown$person_slope_var))
  }
  if (!is.null(result$breakdown$time_intercept_var)) {
    cat("    Raw Time-Level Intercept Variance:\n")
    cat(sprintf("      %.3f\n", result$breakdown$time_intercept_var))
  }
  cat("--------------------------------------------------\n")
}


# Compute correlation matrix from a covariance matrix
calculate_correlation <- function(x) {
  x / sqrt(outer(diag(x), diag(x)))
}

# Compute the intraclass correlation coefficient (ICC)
compute_icc <- function(latent_var_diag, residual_var_diag) {
  latent_var_diag / (latent_var_diag + residual_var_diag)
}

# Compute latent variance via Lyapunov equation for dynamic models using MAP
compute_latent_var_dynamic <- function(est,
                                       beta_prefix,
                                       sigma_eta_prefix,
                                       n_factors,
                                       person_intercept_var_name = NULL,
                                       person_slope_var_name = NULL,
                                       time_slope_var_name = NULL) {
  # --- Compute dynamic latent variance ---
  if (n_factors == 1) {
    slope <- est[beta_prefix, "mean"]
    sigma_eta <- est[sigma_eta_prefix, "mean"]
    # Stationary variance for AR(1): σ² / (1 - φ²)
    sigma_z <- sigma_eta / (1 - slope^2)
  } else {
    # Multivariate case
    mat_a <- matrix(NA, n_factors, n_factors)
    for (i in 1:n_factors) {
      for (j in 1:n_factors) {
        mat_a[i, j] <- est[paste0(beta_prefix, "[", i, ",", j, "]"), "mean"]
      }
    }
    sigma_eta <- matrix(NA, n_factors, n_factors)
    for (i in 1:n_factors) {
      for (j in 1:n_factors) {
        sigma_eta[i, j] <- est[paste0(sigma_eta_prefix,
                                      "[", i, ",", j, "]"), "mean"]
      }
    }
    # Solve Lyapunov equation
    kron_term <- kronecker(mat_a, mat_a)
    vec_sigma_z <- solve(diag(n_factors^2) - kron_term, as.vector(sigma_eta))
    sigma_z <- matrix(vec_sigma_z, nrow = n_factors)
  }
  out <- list(dynamic_var = sigma_z, tot_var = sigma_z)

  # --- Additional variances for random effects ---
  if (!is.null(person_intercept_var_name)) {
    out$person_intercept_var <- setNames(
      sapply(person_intercept_var_name, function(nm) est[nm, "mean"]),
      person_intercept_var_name
    )
    out$tot_var <- out$tot_var + sum(out$person_intercept_var)
  }
  if (!is.null(person_slope_var_name)) {
    out$person_slope_var <- setNames(
      sapply(person_slope_var_name, function(nm) est[nm, "mean"]),
      person_slope_var_name
    )
    out$tot_var <- out$tot_var + sum(out$person_slope_var)
  }
  if (!is.null(time_slope_var_name)) {
    out$time_slope_var <- setNames(
      sapply(time_slope_var_name, function(nm) est[nm, "mean"]),
      time_slope_var_name
    )
    out$tot_var <- out$tot_var + sum(out$time_slope_var)
  }
  return(out)
}

# Compute standardized loadings using MAP estimate
compute_strd_loadings <- function(est,
                                  factor_structure,
                                  var_latent_diag,
                                  sigma_y_prefix,
                                  lambda_prefix) {
  result_list <- list()

  for (i in seq_len(nrow(factor_structure))) {
    row <- factor_structure[i, ]
    ind_idx <- row$indicator
    factor_idx <- row$factor
    loading_id <- row$loading_id
    label <- paste0("indicator_", ind_idx)

    sigma_y_name <- paste0(sigma_y_prefix, "[", ind_idx, "]")
    sigma_y <- est[sigma_y_name, "mean"]

    # Handle fixed loading (e.g., identification constraint)
    if (is.na(loading_id)) {
      lambda <- 1.0
    } else {
      lambda_name <- paste0(lambda_prefix, "[", loading_id, "]")
      lambda <- est[lambda_name, "mean"]
    }

    latent_var <- var_latent_diag[[factor_idx]]
    lambda_std <- lambda * sqrt(latent_var /
                                  (lambda^2 * latent_var + sigma_y))

    result_list[[label]] <- lambda_std
  }

  return(as.data.frame(result_list))
}


# standardization functions for FUSION EXAMPLE
############################
# function that transforms an input covariance matrix x
# into a correlation matrix y (output)
############################
corfun <- function(x){
  y <- x
  for(j in 1:ncol(x)){
    for(k in 1:nrow(x)){
      y[k,j] <- x[k,j]/sqrt(x[k,k]*x[j,j])
    }
  }
  y
}

############################
# Solve discrete Lyapunov equation: covtotal = beta%*%covy%*%beta' + covy
############################
get.cov.from.AR1 <- function(beta, covy) {
  # vec(Sigma_z) = solve(I - A ⊗ A) %*% vec(Q)
  I <- diag(nrow(beta)^2)
  kron_term <- kronecker(beta, beta)
  vec_Sigma_z <- solve(I - kron_term, as.vector(covy))
  matrix(vec_Sigma_z, nrow = nrow(beta))
}

