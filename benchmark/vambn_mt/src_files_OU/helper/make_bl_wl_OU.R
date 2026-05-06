make_bl_wl_OU <- function(data, blname, wlname, out = FALSE, orphans = NA) {
  library(tidyverse)

  vars  <- names(data)
  combs <- expand.grid(from = vars, to = vars)

  # extract visit ID numbers
  combs$visfrom <- gsub(".*_VIS", "", combs$from)
  combs$visto   <- gsub(".*_VIS", "", combs$to)
  combs$fromn <- suppressWarnings(as.numeric(combs$visfrom))
  combs$ton   <- suppressWarnings(as.numeric(combs$visto))

  # type flags
  combs$auxfrom <- grepl("AUX_",     combs$from)
  combs$auxto   <- grepl("AUX_",     combs$to)
  combs$zto     <- grepl("zcode_",   combs$to)
  combs$sfrom   <- grepl("scode_",   combs$from)
  combs$missfrom<- grepl("visitmiss_",combs$from)
  combs$missto  <- grepl("visitmiss_",combs$to)

  # ============== BLACKLIST (forbidden) ==============
  bl <- combs %>%
    filter(
      from == to |                         # no loops
      (!is.na(fromn) & !is.na(ton) & fromn > ton) | # no backward causality
      (auxfrom | auxto) |                 # AUX edges blocked unless whitelisted
      missfrom                            # visitmiss only feeds forward
    )

  # ============== WHITELIST (allowed structural edges) ==============

  # 1) visitmiss_VISt → AUX_VISt  (missing visit explains AUX missing)
  wl_visitmiss_aux <- combs %>%
    filter(missfrom & auxto & fromn == ton)

  # 2) visitmiss_VISt → visitmiss_VIS(t+1)  (dropout propagation)
  wl_miss_chain <- combs %>%
    filter(missfrom & missto & ton == fromn + 1)

  # 3) AUX_VISt → same variable_VISt
  wl_aux_to_var <- combs %>%
    filter(auxfrom & grepl("LONG_", to)) %>%
    filter(gsub("AUX_|zcode_|scode_", "", from) ==
           gsub("_VIS[0-9]+","",to))

  # 4) AUX_VISt → AUX_VIS(t+1)
  wl_aux_chain <- combs %>%
    filter(auxfrom & auxto & ton == fromn + 1)

  # 5) scode_VAR → zcode_VAR (latent alignment)
  wl_s_to_z <- combs %>%
    filter(sfrom & zto) %>%
    filter(gsub("scode_","",from) == gsub("zcode_","",to))

  # Combine whitelist
  wl <- bind_rows(wl_visitmiss_aux, wl_miss_chain, wl_aux_to_var, wl_aux_chain, wl_s_to_z) %>%
        distinct(from,to)

  # Clean BL
  bl <- bl %>% distinct(from,to)

  if(out) return(list(bl=bl, wl=wl))

  write.csv(bl[,c("from","to")], blname,row.names=FALSE)
  write.csv(wl[,c("from","to")], wlname,row.names=FALSE)
  message("✔ OU whitelist/blacklist generated for ", nrow(bl)," BL edges & ", nrow(wl)," WL edges")
}
