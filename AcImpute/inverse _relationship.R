
#Calculate dropout rates for different expression levels in various cell types
data_dir = "E:/学习箱/论文/2/datasets/Usoskin_silver/"
rawdata<-read.csv(paste0(data_dir,"Usoskin_RAW.csv"))
labels = read.table("E:/学习箱/论文/2/datasets/Usoskin_silver/Usoskin_RAW.txt", 
                    skip = 0, nrows = 1, 
                    stringsAsFactors = FALSE)
label <- as.numeric(labels)
datarate <- matrix(nrow = 0, ncol = 3)
for (i in unique(label)) {
  type_col = which(label==i)
  gene_exp = rawdata[,-1]
  gene_exp = gene_exp[,type_col]
  row_mean <- rowMeans(gene_exp)
  zero_proportion  <- rowSums(gene_exp == 0)/ ncol(gene_exp)
  sorted_indices <- order(row_mean)
  row_means_sorted<-sort(row_mean)
  zero_proportion_sorted<-zero_proportion[sorted_indices]
  group_type<-rep(i,length(row_mean))
  merged_data <-cbind(group_type,row_means_sorted,zero_proportion_sorted)
  datarate<-rbind(datarate,merged_data)

}
library(dplyr)
datarate <- data.frame(datarate)
# 使用dplyr进行操作
result <- datarate %>%
  group_by(group_type, row_means_sorted) %>%
  summarize(
    Mean_zero_proportion = mean(zero_proportion_sorted),
    Max_zero_proportion = max(zero_proportion_sorted),
    Min_zero_proportion = min(zero_proportion_sorted),
    countn = n()
  ) %>%
  distinct(group_type, row_means_sorted, .keep_all = TRUE)
result<-data.frame(result[,1:3])
# # 创建折线图
# p<-ggplot(result, aes(x = row_means_sorted, y = Mean_zero_proportion, color = group_type, group = group_type)) +
#   geom_line() +
#   geom_point() +
#   labs(title = "折线图", x = "X轴", y = "Y轴") +
#   scale_color_brewer(palette = "Paired") +
#   theme_minimal()
# print(p)
