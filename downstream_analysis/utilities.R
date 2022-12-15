library(reshape2)
library(ggplot2)
library(ComplexHeatmap)
library(RColorBrewer)
library(viridis)
library(grid)
library(circlize)
library(ggrepel)

#plot settings
source('~/plot/publication_plot_theme.R')
#### some functions

#order features
get_feature_orders=function(topic_matrix,num_topic=100){

  feature_ids=c()#ordered ids
  for (i in 1:num_topic){
    value=topic_matrix[,i]
    feature_ordered=rownames(topic_matrix)[order(value,decreasing = T)]
    feature_ids=cbind(feature_ids,feature_ordered)
  }
  colnames(feature_ids)=paste0('topic',1:num_topic)
  return(feature_ids)
}

##calculate correlation between ranks
cal_corr=function(version,name_index,id_index,rna_topic,adt_topic){
  #read all gene ids and protein ids
  if (strsplit(version,'/')[[1]][1]=='covid'){
    all_genes=read.csv('./data/covid/rna_name.csv',header=T)
    all_proteins=read.csv('./data/covid/protein_name.csv',header=T)
  } else{#nips
    all_genes=read.csv('./data/rna_name.csv',header=T)
    all_proteins=read.csv('./protein_name_complete.csv',header=T)
  }

  #get corresponding ids based on gene name & protein name
  gene_name_id=c()
  for (i in rownames(rna_topic)){
    id=all_genes[all_genes[,name_index]==i,][,id_index]
    if (length(id)!=0){
      for (d in id){#if more than 1 ids
        gene_name_id=rbind(gene_name_id,c(i,d))
      }    
    }
  }
  colnames(gene_name_id)=c('name','id')
  gene_name_id=as.data.frame(gene_name_id)

  protein_name_id=c()
  for (i in rownames(adt_topic)){
    id=all_proteins[all_proteins[,name_index]==i,][,id_index]
    if (length(id)!=0){
    protein_name_id=rbind(protein_name_id,c(i,id))}
  }
  colnames(protein_name_id)=c('name','id')
  protein_name_id=as.data.frame(protein_name_id)

  ### select common ids 
  common_ids=intersect(gene_name_id$id,protein_name_id$id)
  
  ## get corresponding names
  common_pro=c()
  for (i in common_ids){
    common_pro=c(common_pro,protein_name_id[protein_name_id$id==i,]$name)
  }
  
  common_genes=c()
  for (i in common_ids){
    common_genes=c(common_genes,gene_name_id[gene_name_id$id==i,]$name)
  }
  
  ##make sure the same order
  protein_topic_sub=adt_topic[common_pro,]
  rna_topic_sub=rna_topic[common_genes,]

  ## under each topic, get rank, calculate correlation for each topic
  rank_gene=apply(rna_topic_sub,2,rank)#column
  rank_pro=apply(protein_topic_sub,2,rank)#column
  corr=c()
  p=c()#wilcox paired test

  for (i in 1:num_topic){
    x=rank_gene[,i]
    y=rank_pro[,i]
    c=cor(x,y)
    corr=c(corr,c)
    
    w=wilcox.test(x, y, paired = TRUE, alternative = "two.sided")
    p=c(p,w$p.value)
  }
  p_adjust=p.adjust(p,'BH')

  plot_corr=data.frame(cor=corr,p_value=p,p_value_adj=p_adjust)

  return(plot_corr)
}

#plot heatmap for top genes and protein
plot_top_feature_in_selected_topic=function(type,top_feature_num,
                                            all_ids,topic_matrix,
                                            selected_topic,
                                            save_path,name='',
                                            width=5,height=10){
    top=all_ids[1:top_feature_num,selected_topic]
    all_names=melt(top)[,3]#all top gene/protein names

    m=c()#values
    for ( i in selected_topic){
      m=cbind(m,topic_matrix[all_names,i])
    }

    rownames(m)=all_names
    colnames(m)=as.character(selected_topic)

    m_scale=apply(m,2,function(x) x/max(abs(x)))#scale by column(within each topic)
    m_plot=melt(m_scale)
    m_plot$Var2=factor(m_plot$Var2,levels=unique(m_plot$Var2))
    plot=ggplot(m_plot,aes(x=Var2,y=Var1,fill=value))+
         geom_tile() +
         scale_fill_gradient2(low = "blue", mid = "white", high = "red",midpoint = 0)+
         labs(title = "",x='topic',y=type) +
         theme_bw() +
         theme(axis.text.x = element_text(angle = -90, hjust = 0,size=10),
               axis.text.y = element_text(size=10))

    save_name=paste0(save_path,'top',top_feature_num,type,name,'.png')
    ggsave(filename=save_name,plot=plot,dpi=320,width=width,height=height)
}

### plot sample by topic
plot_sample_by_topic=function(plot_cell_topic,version,coll,num_select_topic,save_name,is_cluster=TRUE){
# Create the heatmap annotation
  if (version=='covid'){
    ha <- HeatmapAnnotation(
      cellType=plot_cell_topic$initial_clustering,
      severity=plot_cell_topic$Status_on_day_collection_summary,
      status=plot_cell_topic$Status,
      col = coll)
  } else if(version=='rna_atac'){#atac
    ha <- HeatmapAnnotation(
      cellType=plot_cell_topic$cellType,
      col = coll,
      annotation_name_gp= gpar(fontsize = 20))
    
  } else{
    ha <- HeatmapAnnotation(
      cellType1=plot_cell_topic$cellType1,
      cellType2=plot_cell_topic$cellType2,
      col = coll,
      annotation_name_gp= gpar(fontsize = 20))
  }

  #prepare plot data
  plot_matrix=as.matrix(t(plot_cell_topic[,1:num_select_topic]))
  col_fun = colorRamp2(c(min(plot_matrix), 0, max(plot_matrix)), c("blue", "white", "red"))

  # Combine the heatmap and the annotation
  # !!!!! check dimension of cell_topic_select, make sure all are numeric !!!!
  png(file=save_name,
      width = 1500, height = 1000,units='px',bg = "transparent",res=100)
  h=Heatmap(plot_matrix, col=col_fun,
            show_column_names = FALSE,
            cluster_columns = is_cluster,
            cluster_rows = TRUE,
            top_annotation = ha,
            row_names_gp = grid::gpar(fontsize = 20)
            )
  draw(h)
  dev.off()
}

## plot correlation
plot_correlation=function(corr_plot,save_name){
  plot_corr=ggplot(corr_plot,aes(x=x,y=y,color=color))+
    geom_point(size=2)+
    theme_Publication(base_family='Arial')+
    theme(panel.grid = element_blank(),
          panel.grid.major=element_line(colour=NA),
          legend.position='none',
          axis.title.x=element_text(size=20),
          axis.title.y=element_text(size=20),
          axis.text.y = element_text(size = 20),
          axis.text.x = element_text(size = 20))+
    labs(x='',y='correlation',title='Correlation among genes and proteins')+
    geom_hline(yintercept = 0,linetype = 'dashed')
  ggsave(filename=save_name,plot=plot_corr,dpi=320,width=10,height=3)
}

## plot q values
plot_q_value=function(plot_data,save_name){
  plot_q=ggplot(plot_data,aes(x=x,y=y,color=color,label=label))+
    geom_point(alpha=0.3,size=1.5)+
    #geom_text_repel()+
    theme_Publication(base_family='Arial')+
    theme(panel.grid=element_blank(),
          panel.grid.major=element_line(colour=NA),
          legend.position='none',
          axis.title.x=element_text(size=20),
          axis.title.y=element_text(size=20),
          axis.text.y = element_text(size = 20),
          axis.text.x = element_text(size = 20))+
    labs(x='',y='-ln(q value)',title='')+
    geom_hline(yintercept = 3,linetype = 'dashed',col='red')
  ggsave(filename=save_name,plot=plot_q,dpi=320,width=10,height=3)
}

## plot p values for covid topics
plot_topic_p_value=function(covid_p_plot,save_name){
  plot=ggplot(covid_p_plot,aes(x=topic,y=-log(p++exp(-20)),color=color,label=label))+
    geom_point(size=2)+
    geom_text_repel()+
    theme_Publication(base_family='Arial')+
    theme(panel.grid = element_blank(),
          panel.grid.major=element_line(colour=NA),
          legend.position='none',
          axis.title.x=element_text(size=20),
          axis.title.y=element_text(size=20),
          axis.text.y = element_text(size = 20),
          axis.text.x = element_text(size = 20))+
    labs(x='',y='-log(p)',title='')#+
    #geom_hline(yintercept = 0.05,linetype = 'dashed')
  ggsave(filename=save_name,plot=plot,dpi=320,width=10,height=3)
}
## Differential analysis of topic expression
diff_topic=function(label,cell_info,cell_topic,num_topic=100,alter='greater'){
  label_index=which(names(cell_info)==label)
  label_type=unique(cell_info[,label_index])
  topic_label_p=c()# topic x full_cell_type  matrix storing p values
  topic_label_mean=c() # topic x full_cell_type  matrix storing mean difference 
  for (i in label_type){
    group1=cell_topic[cell_info[,label_index]==i,]#topic values with label
    group2=cell_topic[cell_info[,label_index]!=i,]#topic values without label
    topic_p=c()
    topic_mean=c()
    for (j in 1:num_topic){
      topic_p=c(topic_p,t.test(group1[,j],group2[,j],alternative=alter)$p.value)#upregulated
      topic_mean=c(topic_mean,mean(group1[,j])-mean(group2[,j]))
    }
    #topic_p_adj=p.adjust(topic_p)
    topic_label_p=cbind(topic_label_p,topic_p)
    topic_label_mean=cbind(topic_label_mean,topic_mean)
  }
  #
  topic_label_adj=p.adjust(topic_label_p)
  topic_label_p=matrix(topic_label_adj,nrow=num_topic)
  topic_label_p=as.data.frame(topic_label_p)
  colnames(topic_label_p)=label_type
  #
  topic_label_mean=as.data.frame(topic_label_mean)
  colnames(topic_label_mean)=label_type
  
  return(list('topic_label_p'=topic_label_p,'topic_label_mean'=topic_label_mean))
}

## plot differntially expressed topics 
plot_diff_topic=function(topic_label_list,save_name,width=1000,topic_select=c(1:100)){
  topic_label_mean=as.matrix(topic_label_list$topic_label_mean)
  rownames(topic_label_mean)=1:100
  topic_label_p=as.matrix(topic_label_list$topic_label_p)
  
  topic_label_mean_select=topic_label_mean[topic_select,]
  topic_label_p_select=topic_label_p[topic_select,]
  
  col_fun = colorRamp2(c(min(topic_label_mean_select), 0, max(topic_label_mean_select)), c("blue", "white", "red"))
  #col_fun = colorRamp2(c(-0.6124488, 0, 0.7623149), c("blue", "white", "red"))#confounder same range
  
  png(file=save_name,width = width, height = 1000,units='px',bg = "transparent",res=120)
  h=Heatmap(topic_label_mean_select, col=col_fun,cluster_rows = F,cluster_columns = T,
            row_names_gp = gpar(fontsize = 20),column_names_gp = gpar(fontsize = 20),
          cell_fun = function(j, i, x, y, w, h, fill) {
            if(topic_label_p_select[i, j] < 0.001) {
              gb = textGrob("*")
              gb_w = convertWidth(grobWidth(gb), "mm")
              gb_h = convertHeight(grobHeight(gb), "mm")
              grid.text("*", x, y - gb_h*0.5 + gb_w*0.4,gp = gpar(fontsize = 20))
            } 
          }
          )
  draw(h)
  dev.off()
}
