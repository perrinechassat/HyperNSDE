add_visitmiss_OU <- function(discdata) {
  aux_cols <- grep("^AUX_.*_VIS[0-9]+$", colnames(discdata), value = TRUE)
  
  # extract visit indices automatically → works for VIS00, VIS01,... VIS199+
  visits <- sort(unique(sub("_VIS", "", sub(".*_VIS", "", aux_cols))))

  # 1) Create visitmiss_VISXX for every visit based on AUX==1 across all variables
  for(v in visits){
    cols_v <- grep(paste0("_VIS", v, "$"), aux_cols, value = TRUE)

    if(length(cols_v) > 0){
      discdata[[paste0("visitmiss_VIS", v)]] <- factor(
        apply(discdata[, cols_v, drop=FALSE], 1, function(x) all(x == 1)),
        levels = c(FALSE, TRUE), labels = c(0,1)
      )
    }
  }
  
  # 2) Identify 'orphan AUX': AUX identical to visitmiss → parent should be removed
  rm <- c()
  for(v in visits){
    vm <- paste0("visitmiss_VIS", v)
    cols_v <- grep(paste0("_VIS", v, "$"), aux_cols, value = TRUE)
    
    if(vm %in% colnames(discdata) && length(cols_v)>0){
      for(a in cols_v){
        if(identical(as.numeric(discdata[[a]]), as.numeric(discdata[[vm]]))){
          rm <- c(rm,a)
        }
      }
    }
  }
  
  message("OU visitmiss + orphan AUX summary:")
  message("Number of visits detected: ", length(visits))
  message("AUX orphan variables removed: ", length(rm))

  list(data = discdata, rm = rm)
}
