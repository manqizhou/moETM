### check enriched cell types
### 3 cases: nips_rna+protein, nips_rna+atac, covid

source('./utilities.R')

num_topic=100#as.numeric(args[1])

############################# nips_rna+protein
# read data, change data path accordingly
cell_topic_raw=read.csv('../Result/nips/topic_proportion_rna_protein_before_sm.csv',header=T)#change path
cell_info=cell_topic_raw[,(num_topic+2):dim(cell_topic_raw)[2]]#cell information
cell_topic=cell_topic_raw[,1:(num_topic+1)]
colnames(cell_topic)=c('sample',1:num_topic)

#nips rna+protein
cell_type0=read.csv('../useful_file/cell_type_nips_color.csv',header=T)#use low-res0lution cell type annotation
main_cell_types=c()
all_cell_types=unique(cell_type0$cellType1)
for (i in 1:num_topic){
  t=c()
  for (c in all_cell_types){
    sub_types=cell_type0$cellType2[cell_type0$cellType1==c]
    value=mean(cell_topic[cell_info%in%sub_types,i+1])
    t=c(t,value)
  }
  main_cell_types=rbind(main_cell_types,t)
}
main_cell_types=as.data.frame(main_cell_types)
colnames(main_cell_types)=all_cell_types

enriched=c()
for (i in 1:num_topic){
  cc=names(which.max(main_cell_types[i,]))
  c=c(i,cc,max(main_cell_types[i,]))
  enriched=rbind(enriched,c)
}
enriched=as.data.frame(enriched)
colnames(enriched)=c('topic','enriched','avg_value')

save_name=paste0('../nips_rna+protein_enriched_cell_types.csv')## change save path
write.csv(enriched,file=save_name,quote=F,row.names=F)
