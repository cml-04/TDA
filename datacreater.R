library(scCCESS)
library(SingleCellExperiment)

data(sce, package='scCCESS')
types <- colData(sce)$cellTypes
set.seed(42)

rare_candidates <- c(
  "enterocyte of epithelium of large intestine",  
  "keratinocyte stem cell",                        
  "B cell",                                        
  "immature T cell"                               
)
rare_counts <- c(10, 20, 50)

for (rt in rare_candidates) {
  for (nr in rare_counts) {
    all_of_type <- which(types == rt)
    if (length(all_of_type) < nr) {
      cat("SKIP:", rt, "only has", length(all_of_type), "\n")
      next
    }
    idx_rare <- sample(all_of_type, nr)
    
    other_types <- setdiff(unique(types), rt)
    idx_major <- unlist(lapply(other_types, function(t) {
      sample(which(types == t), 100)
    }))
    
    sce_sub <- sce[, c(idx_major, idx_rare)]
    
    fname <- gsub(" ", "_", rt)
    data_file <- paste0("sce_imbal_", fname, "_n", nr, ".csv")
    label_file <- paste0("labels_imbal_", fname, "_n", nr, ".csv")
    
    write.csv(t(as.matrix(counts(sce_sub))), data_file)
    write.csv(data.frame(
      cell_id = colnames(sce_sub),
      cell_type = colData(sce_sub)$cellTypes
    ), label_file, row.names = FALSE)
    
    cat(rt, "n=", nr, "→", ncol(sce_sub), "cells →", data_file, "\n")
  }
}

