  #伪时间分析
  
  rm(list = ls())
  gc()
  library(monocle)
  library(Seurat)
  library(scran)
  my.monocle <- function(count, cellLabels,cell_type){
    colnames(count) <- 1:ncol(count)
    geneNames <- rownames(count)
    rownames(count) <- 1:nrow(count)
    #构建对象
    pd <- data.frame(timepoint = cellLabels)
    pd <- new("AnnotatedDataFrame", data=pd)
    fd <- data.frame(gene_short_name = geneNames)
    fd <- new("AnnotatedDataFrame", data=fd)
    # dCellData <- scran::newCellDataSet(count, phenoData = pd, featureData = fd, expressionFamily = uninormal())
    
    dCellData <- newCellDataSet(count, phenoData = pd, featureData = fd, expressionFamily = uninormal())
    #过滤基因
    dCellData <- detectGenes(dCellData , min_expr = 0.1)
    expressed_genes <- row.names(subset(fData(dCellData),
                                        num_cells_expressed >= 50))
    #筛选高变异基因
    diff_test_res <- differentialGeneTest(dCellData[expressed_genes,],
                                          fullModelFormulaStr = "~timepoint",
                                          cores = 3)
    ordering_genes <- row.names (subset(diff_test_res, qval < 0.01))
    # 使用 ordering_genes 中的基因来排序 dCellData 中的细胞
    dCellData <- setOrderingFilter(dCellData, ordering_genes)
    #降维以便更好地可视化
    dCellData <- reduceDimension(dCellData, max_components = 2,
                                 method = 'DDRTree', norm_method='none') 
    # orderCells 用于根据降维后的结果对细胞进行排序，推断它们在发育轨迹上的位置
    dCellData <- orderCells(dCellData)
    
    cor.kendall = cor(dCellData@phenoData@data$Pseudotime, as.numeric(dCellData@phenoData@data$timepoint), 
                      method = "kendall", use = "complete.obs")
    
    lpsorder2 = data.frame(sample_name = colnames(count), State= dCellData@phenoData@data$State, 
                           Pseudotime = dCellData@phenoData@data$Pseudotime, rank = rank(dCellData@phenoData@data$Pseudotime))
    
    lpsorder_rank = dplyr::arrange(lpsorder2, rank)
    
    lpsorder_rank$Pseudotime = lpsorder_rank$rank
    
    lpsorder_rank = lpsorder_rank[-4]
    
    lpsorder_rank[1] <- lapply(lpsorder_rank[1], as.character)
    
    subpopulation <- data.frame(cell = colnames(count), sub = as.numeric(cellLabels)-1)
    
    POS <- TSCAN::orderscore(subpopulation, lpsorder_rank)[1]
    
    print(list(cor.kendall=abs(cor.kendall), POS=abs(POS)))
    plot_cell_trajectory(dCellData,color_by = "cell_type",size=1,show_backbone = TRUE)+
       annotate("text", x = -Inf, y = Inf, label = paste0("kendall ",round(abs(cor.kendall),3),"  POS ",round(abs(POS),3)), 
                hjust = 0, vjust = 1, size = 4)+
      theme_minimal() +  # 设置主题
      theme(legend.position = "right")  # 将图例放置在右侧
    
    
    
  }
  
  count<-read.csv("E:/download/GSE75748/GSE75748_sc_time_course_ec.csv",row.names=1)
  data_label<-colnames(count)
  data_label<-as.data.frame(data_label)
  data_label$data_label <- sub(".*[._](.*?)_.*", "\\1", data_label$data_label)
  cellLabels<-data_label
  labelname <- as.data.frame(unique(cellLabels))
  for (i in 1:length(labelname$data_label)) {
    cellLabels$data_label[which(cellLabels$data_label==labelname$data_label[i])]=i
  }
  cellLabels<-cellLabels$data_label
  cell_type<-data_label$data_label
  ###  run Monocle2 
  count<-read.csv("E:/download/GSE75748/AcImpute.csv",row.names=1)
  count<-as.matrix(count)
  lps.monocle <- my.monocle(count, cellLabels,cell_type)
  
  ##############
