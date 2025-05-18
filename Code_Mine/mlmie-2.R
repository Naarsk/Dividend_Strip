################################################################################
# R script for estimating Ridge and LASSO regressions with glmnet
#
# Course: Machine-learning methods in econometrics
# Lecturer: Gautam Tripathi
# Code by: Andreï V. Kostyrka
# Entity: University of Luxembourg
# Date: 2025-03
################################################################################

rm(list = ls())
tryCatch(setwd("~/Dropbox/HSE/15/mlmie/codes/"), error = \(e) return(NULL))

cols5 <- rainbow(5, end = 0.7, v = 0.8)
cols6 <- rainbow(6, end = 0.75, v = 0.8)
data <- read.csv("Data_sa.csv")

i <- 4  # Dependent variable index

data <- as.data.frame(lapply(data, diff)) # Stationarising

matplot(scale(data[, 1:5]),   type = "l", col = cols5, lty = 2 - (1:5 == 4), lwd = (1:5 == 4) + 1, bty = "n", ylab = "", main = "Scaled dependent variables 1-5")
matplot(scale(data[, 6:10]),  type = "l", col = cols5, lty = 1, lwd = 1, bty = "n", ylab = "", main = "Scaled dependent variables 6-10")
matplot(scale(data[, 11:15]), type = "l", col = cols5, lty = 1, lwd = 1, bty = "n", ylab = "", main = "Scaled dependent variables 11-15")

plot(ts(data[, 16:21], end = c(2015, 12), frequency = 12), type = "l", lwd = 2, bty = "n", ylab = "", main = "Explanatory variables 1-6")
plot(ts(data[, 22:26], end = c(2015, 12), frequency = 12), type = "l", lwd = 2, bty = "n", ylab = "", main = "Explanatory variables 7-11")

rownames(data) <- paste0("m", 1:nrow(data))

ny <- 15     # Number of dependent variables

Y <- data[, 1:ny] # Dependent variables
X0 <- data[, (ny+1):ncol(data)] # Explanatory variables
X1 <- rbind(NA, X0[1:(nrow(X0)-1),]) # First lags of regressors
X2 <- rbind(NA, X1[1:(nrow(X1)-1),]) # Second lags of regressors
colnames(X1) <- paste0("l1", colnames(X0))
colnames(X2) <- paste0("l2", colnames(X0))
X <- cbind(X0, X1, X2) # Merge regressors into a single DF

X <- X[3:nrow(X), ] # And leave only complete observations
Y <- Y[3:nrow(Y), ]
# Reorder variables: each X followed by its 2 lags
X <- X[, as.vector(t(matrix(1:ncol(X), ncol = 3)))]

rm(X0, X1, X2, cols5, cols6, ny, data)

# Standardising the variables
Y <- scale(Y)
X <- scale(X)

colnames(Y)[i]

###### LASSO, Ridge,  Elastic Net
library(glmnet)
par(mar = c(2, 2, 0, 0))
plot(as.data.frame(cbind(Y[, i], X[, 1:6])))
dev.off()

X <- as.matrix(X)

# Beginning: L1- and L2-penalised OLS with many penalties
mod1 <- glmnet(x = X, y = Y[, i])
plot(mod1, xvar = "lambda", bty = "n")

mod2 <- glmnet(x = X, y = Y[, i], alpha = 0)
plot(mod2, xvar = "lambda", bty = "n")

mod3 <- glmnet(x = X, y = Y[, i], alpha = 0.4)
plot(mod3, xvar = "lambda", bty = "n")

coef(mod1)
coef(mod1, s = exp(-2))

# Selecting the first 5 variables from the path
nonzero.vars <- rowSums(apply(coef(mod1), 1, function(x) x != 0))
which.has5 <- which(nonzero.vars >= 5+1)[1]
b5 <- coef(mod1)[, which.has5]
b5[b5 != 0]
names(b5[b5 != 0])[-1]

set.seed(1)
mod1.cv <- cv.glmnet(x = X, y = Y[, i])
plot(mod1.cv)

# Why setting seed is important
plot(cv.glmnet(x = X, y = Y[, i]))
plot(cv.glmnet(x = X, y = Y[, i]))
plot(cv.glmnet(x = X, y = Y[, i]))

# Deterministic leave-one-out (LOO) CV -- potentially slow if n>1000
mod1.cvn <- cv.glmnet(x = X, y = Y[, i], nfolds = nrow(Y))
plot(mod1.cvn)
lambda.opt <- mod1.cvn$lambda.min
coef(mod1.cvn, s = "lambda.min")

coef(mod1, s = lambda.opt)
coef(mod1, s = lambda.opt, exact = TRUE, x = X, y = Y[, i])
mod1.fix <- glmnet(x = X, y = Y[, i], lambda = lambda.opt)  # Also correct
coef(mod1.fix)

# Cross-validating Elastic Net alpha (mixing L1 and L2 penalty)
nlambda <- 101  # Determines search thoroughness and plot smoothness
lambda.seq <- exp(seq(-6, 6, length.out = nlambda))
alpha.seq <- seq(0, 1, length.out = 24*30+1)
# Use nlambda = 51, length.out = 21 for reasonable times

elnet.res <- vector("list", length(alpha.seq))
do.plot <- FALSE

tryCatch(load("elnet-res.RData"), error = function(e) return(NULL))
if (!exists("elnet.res")) {
  for (a in seq_along(alpha.seq)) {
    depvar <- colnames(Y)[i]
    mod1 <- glmnet(X, Y[, i], lambda = lambda.seq, alpha = alpha.seq[a])
    mod1cv <- cv.glmnet(X, Y[, i], alpha = alpha.seq[a], lambda = lambda.seq, nfolds = nrow(Y))
    minlambda <- mod1cv$lambda.min
    print(paste0("Ran ElNet for ", depvar, " and alpha=", round(alpha.seq[a], 3), " in ", nrow(Y), " folds"))
    if (do.plot) png(paste0("/tmp/path-a", sprintf("%03d", a), ".png"), 960, 480, pointsize = 16)
    plot(mod1, xvar = "lambda", bty = "n", lwd = 2)
    abline(v = log(mod1cv$lambda.min), lty = 2)
    legend("topright", paste0("alpha=", sprintf("%1.3f", alpha.seq[a])))
    if (do.plot) dev.off()
    
    b <- as.matrix(coef(mod1, s = mod1cv$lambda.min, exact = TRUE, x = X, y = Y[, i]))
    selected <- row.names(b)[b != 0]
    selected <- selected[selected != "(Intercept)"]
    if (length(selected) < 3) { # If the model is so poor, force at least 3 variables
      coefmat <- as.matrix(coef(mod1))
      minlambda <- mod1$lambda[which(colSums(coefmat != 0) >= 4)[1]] # Take the 4 last non-zero regressors (incl. constant)
      b <- as.matrix(coef(mod1, s = minlambda))
      selected <- row.names(b)[b!=0]
      selected <- selected[selected != "(Intercept)"]
    }
    elnet.res[[a]] <- list(selected = selected, mse = mod1cv$cvm)
  }
  save(elnet.res, file = "elnet-res.RData", compress = "xz")
}

sapply(lapply(elnet.res, "[[", "mse"), length)
cvm.mat <- do.call(rbind, lapply(elnet.res, "[[", "mse"))
rownames(cvm.mat) <- paste0("a", alpha.seq)
cvm.mat <- sqrt(cvm.mat)[, ncol(cvm.mat):1]
linds <- 1:ncol(cvm.mat)
opt.ind <- which(cvm.mat == min(cvm.mat), arr.ind = TRUE)
alpha.opt <- alpha.seq[opt.ind[1]]
levs <- round(seq(quantile(cvm.mat, 0.05), quantile(cvm.mat, 0.95), length.out = 19), 3)
contour(alpha.seq, log(lambda.seq), cvm.mat, levels = levs,
        xlab = "alpha", ylab = "Log-lambda", bty = "n",
        main = "Cross-validation of lambda and alpha")
points(alpha.opt, log(lambda.seq)[opt.ind[2]], pch = 16)

# Visualising the 2-parameter model selection
# The lambda sequence should be provided manually!
# Brighter = lower MSE
image(alpha.seq, log(lambda.seq), cvm.mat,
      main = "Heat map of OOS RMSE (darker = higher)",
      xlab = "alpha", ylab = "Ordered lambda", bty = "n")
image(alpha.seq, log(lambda.seq), log1p(cvm.mat - min(cvm.mat)),
      main = "Heat map of OOS RMSE (darker = higher)",
      xlab = "alpha", ylab = "Ordered lambda", bty = "n")


if (do.plot) {
  # Export a GIF:
  # system('magick -delay 10 -loop 0 -verbose path-a*.png path.gif')
  # system('gifsicle -i path.gif -O3 --colors 64 -f --verbose -o path-anim.gif')
  # Export an MP4:
  system('ffmpeg -y -framerate 24 -i /tmp/path-a%03d.png -vcodec libx264 -crf 22 -preset veryslow path.mp4')
  # Delete the newly generated PNG files irreversibly
  unlink("/tmp/path-a*.png")
}



# Relaxing the LASSO
mod1.relax <- glmnet(x = X, y = Y[, i], alpha = 1, relax = TRUE)
plot(mod1.relax, xvar = "lambda", bty = "n")  # Nothing changes; gamma = 1; no relaxation
plot(mod1.relax, gamma = 0.75, xvar = "lambda", bty = "n")
plot(mod1.relax, gamma = 0.5, xvar = "lambda", bty = "n")
plot(mod1.relax, gamma = 0, xvar = "lambda", bty = "n")

mod1.relax.cv <- cv.glmnet(x = X, y = Y[, i], alpha = 1, relax = TRUE, nfolds = nrow(Y))
plot(mod1.relax.cv)
print(mod1.relax.cv)

# Relaxed lasso
b.rl <- coef(mod1.relax, s = mod1.relax.cv$lambda.min, exact = TRUE)
nz.coefs <- setdiff(rownames(as.matrix(b.rl))[as.numeric(b.rl) != 0], "(Intercept)")
frml <- paste0(colnames(Y)[i], " ~ ", paste0(nz.coefs, collapse = " + "))
mod.ls <- lm(frml, data = data.frame(X, Y))
lmtest::coeftest(mod.ls, vcov. = sandwich::vcovHC)

