#AcImpute
bexpr = readRDS('E:/学习箱/论文/2/datasets/cellbench/cellbench/GSE86337_processed_count_average_replicates.rds')
count_path = "E:/学习箱/论文/2/datasets/cellbench/sc_10x_5cl/genebycell.csv"
allmtd = list.files('E:/学习箱/论文/2/datasets/cellbench/sc_10x_5cl/methods/')
raw = read.csv(count_path)
for (mtd in allmtd){
  print(mtd)
  sexpr = readRDS(paste0('E:/学习箱/论文/2/datasets/cellbench/sc_10x_5cl/methods/',mtd,'/sc_10x_5cl.rds'))
  if (mtd=='AcImpute'){
    rownames(sexpr) <- sexpr[, 1]
    sexpr<- sexpr[, -1]
  }
  colnames(sexpr)=colnames(raw)
  ct = sub('.*:','',colnames(sexpr))
  res <- lapply(colnames(bexpr), function(cl){
    print(cl)
    be <- bexpr[,cl]
    intgene <- intersect(row.names(sexpr),names(be))
    print(length(intgene))
    tmp=apply(sexpr[intgene, ct[grepl(cl,ct)]],2,cor,be[intgene],method='spearman')
    return(tmp)
  })
  names(res) <- colnames(bexpr)
  View(res)
  saveRDS(res,file=paste0('E:/学习箱/论文/2/datasets/cellbench/sc_10x_5cl/cor/',mtd,'.rds'))
}

allf <- list.files('E:/学习箱/论文/2/datasets/cellbench/sc_10x_5cl/cor')
res <- lapply(allf,function(f) {
  tmp <- readRDS(paste0('E:/学习箱/论文/2/datasets/cellbench/sc_10x_5cl/cor/',f))
 })

names(res) <- sub('.rds','',allf)
cn <- table(unlist(sapply(res,names)))
cn <- names(cn)[cn==length(res)]

res <- sapply(res,function(i) i[cn])

res <- apply(res, 2, unlist)
library(ggplot2)
library(reshape2)
mp <- apply(res,2,median,na.rm=T)
pd <- melt(res)
colnames(pd) <- c('cell','Method',"Correlation")
pd[,'Method'] <- factor(as.character(pd[,'Method']),levels = c('raw',setdiff(names(sort(mp,decreasing = T)),'raw')))
# install.packages("ggpubr")
library(ggpubr)
########
#分类型
type5<-c("A549","HCC827","H1975","H2228","H838")
pd[, 1]=gsub("\\..*", "", pd[, 1])

pd=pd[pd$Method%in%c("AcImpute","raw"),]
pd$Method <- factor(pd$Method, levels =c("AcImpute","raw"))
par(font.axis = 200)
# 绘制小提琴图和箱线图
p <- ggplot(pd, aes(x = cell, y = Correlation, fill = Method)) +
  geom_violin(position = position_dodge(0.8), width = 0.8,alpha=0.5) +
  geom_boxplot(width = 0.2, position = position_dodge(0.8), color = "black") +
  labs(title = "sc_10x_5cl 3918cell×10164genes", x = "Type", y = "Correlation") +
  theme_minimal() +
  theme(
    panel.background = element_blank(),  # 取消面板背景
    panel.grid.major = element_blank(),  # 取消主要网格线
    panel.grid.minor = element_blank(),  # 取消次要网格线
    axis.text.x = element_text(size = 12,color = "black"),
    axis.text.y = element_text(size = 12,color = "black"),
    axis.line = element_line(color = "black")  # 添加坐标轴边线，颜色可以根据需要调整
  )+
  theme(legend.position = "top")+
  stat_compare_means(aes(group = Method),method = 't.test',label = 'p.signif')
p
ggsave(p,filename = "cor.pdf",width = 12,height = 9)

########################################################
#箱线图
###Ziegenhain
rm(list = ls())
gc()
library(Rtsne)
library(ggplot2)
library(parallel)
library(ClusterR)
library(tidyr)
library(dplyr)
library(gridExtra)
library(grid)
library(kernlab)
library(RColorBrewer)
library(ggeasy)   #easy_add_legend_title
library(patchwork)  #p1+p2
library(ggsignif)  #箱线图p值
library(ggpubr)#箱线图p值

corercc <-function(ercc,methodss){
  corpeason<-c()
  for (i in 1:dim(methodss)[2]) {
    corpeason<-c(corpeason,cor(ercc,methodss[,i]))
  }
  return(corpeason)
}

library("xlsx")

dataercclong<-c()
datasets<-c("SmartSeq2","CELseq2","MARSseq","SCRBseq","SmartSeq")
for (i in 1:length(datasets)) {
  eval(parse(text = paste0("AcImpute<-read.csv('E:/学习箱/论文/2/code/data/",datasets[i],"_AcImpute.csv',header = T)")))
   eval(parse(text = paste0(datasets[i],"rawdata<-read.csv('E:/学习箱/论文/2/datasets/Ziegenhain/",datasets[i],".csv',header = T)")))
  eval(parse(text = paste0("erccrawloc = grep('ERCC',",datasets[i],"rawdata[,1])")))
  eval(parse(text = paste0("erccraw = ",datasets[i],"rawdata[erccrawloc,]")))
  ercc<-read.xlsx("E:/学习箱/论文/2/datasets/Ziegenhain/ERCC.xlsx",1)
  ercc = as.matrix(ercc[,c(2,4)])
  
  erccAcImpute = AcImpute[grep('ERCC',AcImpute[,1]),]
  common_rows <- intersect(erccraw[, 1], erccAcImpute[, 1])
  
  erccAcImpute = erccAcImpute[erccAcImpute[,1]%in%common_rows,]
  erccraw = erccraw[erccraw[,1]%in%common_rows,]
  ercc = ercc[ercc[,1]%in%gsub("g", "", common_rows),]
  erccAcImpute = erccAcImpute[order(erccAcImpute[,1]),]
  erccraw = erccraw[order(erccraw[,1]),]
  ercc = ercc[order(ercc[,1]),]
  
  erccAcImpute  = apply(erccAcImpute[,-1],2,as.numeric)
  ercc = as.numeric(ercc[,2])
  
  erccrawcor = corercc(log(ercc),log(erccraw[,-1]+1))
  erccAcImpute = corercc(log(ercc),log(erccAcImpute+1))
  
  dataercc=cbind(erccrawcor,erccAcImpute)
  dataercc=data.frame(dataercc)
  colnames(dataercc)<-c("raw","AcImpute")
  dataercc <- dataercc %>% 
    gather(key = 'Ziegenhain',value = 'correlation') #命名两个列名
  eval(parse(text = paste0("methods<-rep('",datasets[i],"',length(erccrawcor))")))
  dataercctmp = cbind(methods,dataercc)
  dataercclong = rbind(dataercclong,dataercctmp)
  #内容没有 格式转换
  # InsectSprays %>% {
  eval(parse(text = paste0("p",datasets[i],"<- ggplot(dataercc,mapping = aes(x = '',y = correlation , fill = factor(Ziegenhain,levels = c('raw','AcImpute')) ))+
     geom_boxplot(outlier.shape = 21,outlier.colour = 'red',outlier.fill ='blue',
                                      col = 'black',width=1.2)+#,fill = brewer.pal(7,'Paired')
     theme_bw()+
     # theme(axis.text.x = element_text(angle = 45,size = 10, color = 'black',hjust = 1,vjust = 1))+##设置x轴字体大小
     labs(title = '",datasets[i],"',x = ' ')+
     easy_add_legend_title('Methods')+
      stat_compare_means(method = 't.test')")))
  
}

pSmartSeq2+pCELseq2+pSmartSeq+pSCRBseq+pMARSseq+guide_area()+ plot_layout(guides = "collect")
library(ggpubr)
#分组绘图
p<-ggboxplot(dataercclong,x="methods",y="correlation",fill ="Ziegenhain",palette = 'mycol')
ercclong<-p+stat_compare_means(aes(group = Ziegenhain),method = 't.test')+
  easy_add_legend_title('Methods')
# write.csv(dataercclong,"D:/software/RStudio/single_cell/datasets/Ziegenhain/result/dataercclong.csv")
# ggsave(ercclong,filename = "D:/software/RStudio/single_cell/datasets/Ziegenhain/result/ercclonggroup.pdf",width = 12,height = 9)

