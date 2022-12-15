## this script is used to plot correlation, q values, top features

source('./publication_plot_theme.R')
source('./utilities.R')
version='covid'
num_topic=100
########### read correlation data, GSEA q values data
save_path=paste0('../plot/',version,'/')

plot_corr=read.csv(paste0('../plot/',version,'/plot_correlation.csv'))
q_rna=read.csv(paste0('../plot/',version,'/q_values_c7_rna.csv'))
q_protein=read.csv(paste0('../plot/',version,'/q_values_c7_protein.csv'))

corr_color=rep(NA,num_topic)
corr_color[plot_corr$cor>=0]='red'
corr_color[plot_corr$cor<0]='blue'
corr_plot=data.frame(x=1:num_topic,y=plot_corr$cor,color=corr_color)

q_rna_plot=data.frame(x=q_rna$x,y=-log(q_rna$y+exp(-20)),color=q_rna$colorr)
q_rna_plot$label=ifelse(q_rna_plot$y>(-log(0.01)),q_rna_plot$x,'')

q_protein_plot=data.frame(x=q_protein$x,y=-log(q_protein$y+exp(-20)),color=q_protein$colorr)
q_protein_plot$label=ifelse(q_protein_plot$y>(-log(0.01)),q_protein_plot$x,'')

###################### plot_corr
save_name=paste0(save_path,'gene_protein_corr.png')
plot_correlation(corr_plot,save_name)

#####################  plot q value for genes
save_name=paste0(save_path,'gene_gsea_q_value_rna.png')
plot_q_value(q_rna_plot,save_name)

######################  plot q values for proteins
save_name=paste0(save_path,'protein_gsea_q_value_atac.png')
plot_q_value(q_protein_plot,save_name)

######################  plot p values for differentially expressed topics
topic_covid_p=read.csv(paste0('../plot/',version,'/topic_covid_p.csv'))
topic_severity_p=read.csv(paste0('../plot/',version,'/topic_severity_p.csv'))
##
covid_p_plot=data.frame(topic=1:num_topic,p=topic_covid_p$Covid,
                        color=ifelse(topic_covid_p$Covid<0.001,'#4DBBD5FF','#F39B7FFF'))
covid_p_plot$label=ifelse(covid_p_plot$p<0.001,covid_p_plot$topic,'')
save_name=paste0(save_path,'covid_topic_p_value.png')
plot_topic_p_value(covid_p_plot,save_name)
##severity
severity_p_plot=data.frame(topic=1:num_topic,
                           p=topic_severity_p$Critical,
                           color=ifelse(topic_severity_p$Critical<0.001,'#4DBBD5FF','#F39B7FFF'))
severity_p_plot$label=ifelse(severity_p_plot$p<0.001,severity_p_plot$topic,'')
save_name=paste0(save_path,'critical_topic_p_value.png')
plot_topic_p_value(severity_p_plot,save_name)

#######################  plot sample by topic heatmap
type='rna_protein'
is_plot=F
is_diff_topic=T
is_all=F
topic_select='48,78,5,80,83,98,4,23,99,47,21,85,24,63,6,55,22,86,31,39,69,77,26,74,91,42,87'
args=c(num_topic,version,type,topic_select,is_plot,is_diff_topic,is_all)
system(paste(c('Rscript','./plot_sample_by_topic.R',args),collapse=' '))


#######################  plot top features
#read gene by topic matrix, and all gene names
rna_topic=read.csv(paste0('../Result/',version,'/topic_rna.csv'),header=T)
rna_names=rna_topic[,1]
rna_topic=rna_topic[,-1]
rownames(rna_topic)=rna_names
colnames(rna_topic)=paste0('topic',1:num_topic)

#read protein by topic matrix, and all protein names
adt_topic=read.csv(paste0('../Result/',version,'/topic_protein.csv'),header=T)
protein_names=adt_topic[,1]
adt_topic=adt_topic[,-1]

if (version=='covid'){  #remove AB_ in covid protein names. Optional data preparation process
  protein_names_new=c()
  for (i in protein_names){
    protein_names_new=c(protein_names_new,paste0(strsplit(i,'AB_')[[1]][-1]))
  }
  rownames(adt_topic)=protein_names_new
} else{
  rownames(adt_topic)=protein_names
}

colnames(adt_topic)=paste0('topic',1:num_topic)

#####select 10 example topics
## order features
gene_ids=get_feature_orders(rna_topic,num_topic=num_topic)
protein_ids=get_feature_orders(adt_topic,num_topic=num_topic)


topic_select=c(4,5,24,31,42,74,76,85,86,91)
top_feature_num=5
save_path=paste0('../plot/',version,'/')

plot_top_feature_in_selected_topic('gene',top_feature_num,gene_ids,protein_ids,rna_topic,adt_topic,topic_select,save_path,'select')
plot_top_feature_in_selected_topic('protein',top_feature_num,gene_ids,protein_ids,rna_topic,adt_topic,topic_select,save_path,'select')



#######check if top features are cell marker
ref=read.table('../useful_file/Human_cell_markers.txt',sep='\t',header=T)
#sort(unique(ref$cellName))#check match names first
cell_name='CD8+ T cell'#

a=ref[ref$cellName%in%cell_name,c(8,9,11)]
markers=c()
for (i in 1:dim(a)[1]){
  for (j in 1:dim(a)[2]){
    b=strsplit(a[i,j],', ')[[1]]
    markers=c(markers,b)
  }
}
markers=unique(markers)

g=gene_ids[1:5,50]#topic num
p=simu_gene_ids[1:5,50]
## check if top features are biomarkers
g %in% markers
p %in% markers


