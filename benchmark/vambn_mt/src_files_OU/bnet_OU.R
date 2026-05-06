############# bnet_OU.R
# This file runs the Bayesian network for the OU dataset
# and generates virtual patients (VPs) + VP missingness lists
############################################################

rm(list = ls())

############################
### Libraries & helpers  ###
############################

library(tidyverse)
library(arules)
library(mclust)
library(rpart)
library(bnlearn)
library(parallel)

# general helpers
source('helper/clean_help.R')        # includeVar()
source("helper/simulate_VP.R")       # simulate_VPs()
source("helper/save_VPmisslist.R")    # save_VPmisslist()

# OU-specific helpers
source("helper/merge_data_OU.R")       # merge_data_OU()
source("helper/addnoise.R")            # addnoise()
source("helper/add_visitmiss_OU.R")    # add_visitmiss_OU()
source("helper/make_bl_wl_OU.R")       # make_bl_wl_OU()


############################g
### Settings & data prep ###
############################

# Name prefix for output files
name      <- "OU"
data_out  <- paste0("../data_preprocessing/data/data_full_out/", name)
scr       <- "bic-cg"   # BN score 
mth       <- "mle-cg"      # BN parameter learning method

cat(">>> Merging data (latent + longitudinal + static + AUX) via merge_data_OU()...\n")
data <- merge_data_OU()   # returns full merged table, already saved as data_final.rds

# Keep subject IDs separately
pt <- data$SUBJID
data$SUBJID <- NULL

cat(">>> Adding noise to imputed / constant regions (addnoise)...\n")
discdata <- addnoise(data, noise = 0.01)

cat(">>> Building visitmiss variables and identifying orphan AUX (add_visitmiss_OU)...\n")
vm_out   <- add_visitmiss_OU(discdata)
discdata <- vm_out$data
rm_cols  <- vm_out$rm

# Remove AUX columns that are identical to visitmiss parents
if (length(rm_cols) > 0) {
  cat("Removing orphan AUX columns:\n")
  print(rm_cols)
  discdata <- discdata[, !(names(discdata) %in% rm_cols)]
}

# Remove AUX that are almost always 0 / trivial
cat(">>> Removing low-information AUX columns...\n")
lowaux <- discdata[, grepl("AUX_", colnames(discdata)) &
                      !(colnames(discdata) %in% rm_cols)]

if (ncol(lowaux) > 0) {
  to_drop <- sapply(colnames(lowaux), function(x) {
    # sum of 1's <= 1 → almost never indicates missingness
    sum(as.numeric(as.character(lowaux[, x]))) <= 1
  })
  lowaux_names <- colnames(lowaux)[to_drop]
  if (length(lowaux_names) > 0) {
    cat("Dropping low-utility AUX columns:\n")
    print(lowaux_names)
    discdata <- discdata[, !(names(discdata) %in% lowaux_names)]
  }
}

# Orphan node names for make_bl_wl_OU (same idea as DONALD)
orphans <- gsub("AUX_", "", rm_cols)
orphans <- unname(sapply(
  orphans,
  function(x) ifelse(!grepl("SA_", x), paste0("zcode_", x), x)
))


############################
### Bayesian network     ###
############################

cat(">>> Preparing blacklist/whitelist...\n")

# Save list of variable names
datan <- names(discdata)
write.csv(datan, "../data_preprocessing/data/data_full_out/data_names.csv", row.names = FALSE)

blname <- paste0(data_out, "_bl.csv")
wlname <- paste0(data_out, "_wl.csv")

# Create blacklist/whitelist for OU data
# make_bl_wl_OU(discdata, blname, wlname, out = FALSE, orphans = orphans)

# Read black/white lists
bl <- read.csv(blname, stringsAsFactors = FALSE)
wl <- read.csv(wlname, stringsAsFactors = FALSE)

cat("\n>>> Filling remaining NA values for BN compatibility...\n")

for (col in names(discdata)) {

  # numeric → mean + jitter
  if (is.numeric(discdata[[col]])) {
    na_idx <- is.na(discdata[[col]])
    if (any(na_idx)) {
      mu <- mean(discdata[[col]], na.rm=TRUE)
      sdv <- sd(discdata[[col]], na.rm=TRUE)
      if (is.na(sdv) || sdv == 0) sdv <- abs(mu*0.001) + 1e-6
      discdata[[col]][na_idx] <- mu + rnorm(sum(na_idx), 0, sdv*0.001)
    }
  }

  # categorical → replace NA with mode
  else {
    if (any(is.na(discdata[[col]]))) {
      mode_val <- names(which.max(table(discdata[[col]])))
      discdata[[col]][is.na(discdata[[col]])] <- mode_val
    }
  }
}

cat(">>> Remaining NA count after patch = ", sum(is.na(discdata)), "\n")


bn_file <- paste0(data_out, "_finalBN.rds")

if (file.exists(bn_file)) {
  cat(">>> finalBN already exists — loading...\n")
  finalBN <- readRDS(bn_file)
} else {
  cat(">>> Learning final BN structure...\n")
  finalBN <- tabu(
    discdata,
    maxp      = 5,
    blacklist = bl,
    whitelist = wl,
    score     = scr
  )
  saveRDS(finalBN, bn_file)
}
cat("TABU DONE\n")


boot_file <- paste0(data_out, "_bootBN.rds")

if (file.exists(boot_file)) {
  cat(">>> Bootstrapped network already exists — loading...\n")
  boot.stren <- readRDS(boot_file)
} else {
  cat(">>> Bootstrapping network — this is long...\n")
  cores <- parallel::detectCores()
  cl <- parallel::makeCluster(cores)

  boot.stren <- boot.strength(
    discdata,
    algorithm      = "tabu",
    R              = 200,
    algorithm.args = list(maxp = 5, blacklist = bl, whitelist = wl, score = scr),
    cluster        = cl
  )

  parallel::stopCluster(cl)
  saveRDS(boot.stren, boot_file)
}
cat("BOOT DONE\n")



fit_file <- paste0(data_out, "_finalBN_fitted.rds")

if (file.exists(fit_file)) {
  cat(">>> Loading fitted BN...\n")
  fitted <- readRDS(fit_file)
} else {
  cat(">>> Fitting BN parameters...\n")
  fitted <- bn.fit(finalBN, discdata, method = mth)
  saveRDS(fitted, fit_file)
}
cat("FIT DONE\n")


############################
### Virtual patients (VP) ###
############################

cat(">>> Simulating virtual patients with simulate_VPs()...\n")
n_real <- nrow(real)
virtual <- simulate_VPs(
  real,
  finalBN,
  fitted,
  iterative = FALSE,
  scr,
  mth,
  wl,
  bl,
  n = n_real      # same number of VPs as real patients
)


############################
### Save outputs          ###
############################

cat(">>> Saving real & virtual patient datasets...\n")

# attach SUBJID back to real
real$SUBJID <- pt

saveRDS(virtual, paste0(data_out, "_VirtualPPts.rds"))
write.csv(virtual, paste0(data_out, "_VirtualPPts.csv"), row.names = FALSE)

saveRDS(real, paste0(data_out, "_RealPPts.rds"))
real$SUBJID <- NULL
write.csv(real, paste0(data_out, "_RealPPts.csv"), row.names = FALSE)

cat(">>> Saving VP missingness list for HI-VAE decoding...\n")
save_VPmisslist(virtual, "")

cat("BNet OU finished.\n")
