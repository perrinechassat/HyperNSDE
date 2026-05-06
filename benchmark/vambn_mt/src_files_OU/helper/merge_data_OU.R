############# merge_data_OU.R
# Build final dataset for Bayesian Network from:
#  - Imputed data from HI-VAE step 1
#  - Auxiliary missingness indicators
#  - Latent embeddings (HI-VAE encoded)
############################################

rm(list=ls())
library(tidyverse)
source("helper/clean_help.R")   # includeVar()

merge_data_OU <- function(){


    data_all <- readRDS("../data_preprocessing/data/data_full_out/data_all_imp.rds")     # produced by impute_aux_OU.R
    data_aux <- readRDS("../data_preprocessing/data/data_full_out/data_aux.rds")         # missing indicators
    data_meta <- read.csv("metaenc.csv")               # produced by step2_HIVAE

    # Merge longitudinal + static blocks
    message("\n--- Merging longitudinal VIS data ---")

    long_blocks <- names(data_all)[grepl("^LONG_", names(data_all))]
    data_long <- reduce(lapply(long_blocks, function(x) data_all[[x]]),
                        merge, by="SUBJID")

    # Merge static
    message("--- Merging static (stalone_VIS00) ---")
    data_stalone <- data_all[["stalone_VIS00"]]

    # Convert AUX list to single table
    message("--- Merging AUX missingness masks ---")
    data_aux <- reduce(data_aux, merge, by="SUBJID")
    data_aux <- as.data.frame(lapply(data_aux, factor))  # AUX must remain categorical


    if(!"SUBJID" %in% colnames(data_meta)){
        message("⚠ metaenc.csv has no SUBJID — assigning subject index as ID")

        data_meta$SUBJID <- factor(seq_len(nrow(data_meta)))
        
        # OPTIONAL: verify that number of subjects matches
        if(nrow(data_meta) != nrow(data_stalone)){
            stop("metaenc rows do not match number of subjects in stalone_VIS00 !
                meta_rows=",nrow(data_meta)," vs SUBJ=",nrow(data_stalone))
        }
    }

    # Merge ALL components: (Long + static + AUX + latent encoders)
    message("\n--- Final Merge ---")
    data <- reduce(list(data_meta, data_long, data_stalone, data_aux),
                   merge, by="SUBJID")

    # Remove constant / zero variance variables
    drop <- colnames(data)[-includeVar(data)]
    if(length(drop) > 0){
        message(" Removing ", length(drop), " constant/no-information columns")
        print(drop)
        data <- data[ , includeVar(data)]
    }

    # cleanup + save
    data$SUBJID <- factor(data$SUBJID)

    # Convert latent embeddings from integer → numeric
    latent_cols <- grep("scode_|zcode_", colnames(data), value=TRUE)
    if (length(latent_cols) > 0) {
      message("Converting ",length(latent_cols)," latent HI-VAE embeddings to numeric...")
      data[latent_cols] <- lapply(data[latent_cols], function(x) as.numeric(as.character(x)))
    }

    message("\n--- Saving final merged dataset ---")
    saveRDS(data, "../data_preprocessing/data/data_full_out/data_final.rds")

    message("\n✔ merge_data_OU completed successfully")
    return(data)
}

