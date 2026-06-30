# End-to-end numerical comparison: coefficients, J-stat, vcov diagonal, weights
library(CBPS)

full <- read.csv("CBPS_python/cbps/data/LaLonde.csv")
nsw_ctrl <- full[full$exper == 1 & full$treat == 0, ]
psid <- full[full$exper == 0, ]
nsw_ctrl$select <- 1
psid$select <- 0
combined <- rbind(nsw_ctrl, psid)

# Use Linear/CBPS2 (both converge, most reliable comparison)
formula <- select ~ age + educ + black + hisp + married + nodegr + re74 + re75
fit <- suppressWarnings(CBPS(formula, data=combined, ATT=0, method="over", twostep=TRUE))

cat("=== Linear/CBPS2 (converged) ===\n")
cat("Coefficients:", paste(sprintf("%.10e", coef(fit)), collapse=", "), "\n")
cat("J-stat:", sprintf("%.10e", fit$J), "\n")
cat("mle.J:", sprintf("%.10e", fit$mle.J), "\n")
cat("Deviance:", sprintf("%.10e", fit$deviance), "\n")
cat("Vcov diagonal:", paste(sprintf("%.10e", diag(fit$var)), collapse=", "), "\n")
cat("Weight sum:", sprintf("%.10e", sum(fit$weights)), "\n")
cat("Weight range:", sprintf("%.10e %.10e", min(fit$weights), max(fit$weights)), "\n")
cat("PS range:", sprintf("%.10e %.10e", min(fit$fitted.values), max(fit$fitted.values)), "\n")

# Also SmithTodd/CBPS1 (converged)
combined$age_sq <- combined$age^2
combined$educ_sq <- combined$educ^2
combined$re75_sq <- combined$re75^2
combined$re74_sq <- combined$re74^2
combined$hisp_re74zero <- combined$hisp * as.numeric(combined$re74 == 0)

formula2 <- select ~ age + educ + black + hisp + married + nodegr + re74 + re75 + age_sq + educ_sq + re74_sq + re75_sq + hisp_re74zero
fit2 <- suppressWarnings(CBPS(formula2, data=combined, ATT=0, method="exact", twostep=TRUE))

cat("\n=== SmithTodd/CBPS1 (converged) ===\n")
cat("Coefficients:", paste(sprintf("%.10e", coef(fit2)), collapse=", "), "\n")
cat("J-stat:", sprintf("%.10e", fit2$J), "\n")
cat("mle.J:", sprintf("%.10e", fit2$mle.J), "\n")
cat("Deviance:", sprintf("%.10e", fit2$deviance), "\n")
cat("Vcov diagonal:", paste(sprintf("%.10e", diag(fit2$var)), collapse=", "), "\n")
cat("Weight sum:", sprintf("%.10e", sum(fit2$weights)), "\n")
cat("Weight range:", sprintf("%.10e %.10e", min(fit2$weights), max(fit2$weights)), "\n")
