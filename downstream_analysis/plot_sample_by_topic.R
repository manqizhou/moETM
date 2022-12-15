## this script is used to plot sample by topic matrix

source('./utilities.R')

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 8) {
  print(length(args))
  stop(" Usage: plot_sample_by_topic.R <num_topic> <version> <type> <topic_select> <is_plot> <is_diff_topic> <is_all>", call.=FALSE)
}

num_topic=as.numeric(args[1])
version=args[2]
type=args[3]#rna_protein or rna_atac
topic_select=as.numeric(strsplit(args[4],',')[[1]])
if (is.na(topic_select)){topic_select=c()}
is_plot=args[5]=='TRUE'
is_diff_topic=args[6]=='TRUE'
is_all=args[7]=='TRUE'#cutoff=0, plot all topics


################## cell by topic matrix
## read data
cell_topic_raw=read.csv(paste0('../Result/',version,'/topic_proportion_',type,'_before_sm.csv'),header=T)

cell_info=cell_topic_raw[,102:dim(cell_topic_raw)[2]]#cell information
cell_topic=cell_topic_raw[,1:101]
colnames(cell_topic)=c('sample',1:num_topic)

###### plot
if (is_plot==T){
  #for visualiztion, subsample 
  #rows: subsample 10000
  #columns: select topics with sum(abs) > round 3st quantile
  sub_num=10000#subsample 10000 cells
  sub_ind=sample(1:dim(cell_topic)[1],sub_num,rep=F)
  
  col_sum=colSums(abs(cell_topic[,2:(num_topic+1)]))
  if (is_all){#plot all topics
    cutoff=0
  } else{#around 3/4q of col_sum
    cutoff=round(as.numeric(summary(col_sum)["3rd Qu."]))
  }
  cell_topic_select=cell_topic[sub_ind,2:(num_topic+1)][,col_sum>cutoff]
  
  #add manually selected topics
  if (length(topic_select)>0){
    old_col=colnames(cell_topic_select)
    add_col=c()
    for (i in topic_select){
      if (!(i %in% colnames(cell_topic_select))){
        cell_topic_select=cbind(cell_topic_select,cell_topic[sub_ind,i+1])#cell_topic's 1st row is sample id
        add_col=c(add_col,i)
      }
    }
    colnames(cell_topic_select)=c(old_col,add_col)
  }
  num_select_topic=dim(cell_topic_select)[2]

  ### add annotation color
  ## for covid
  manualcolors<-c('forestgreen', 'orange', 'cornflowerblue', 'darkolivegreen4', 'indianred1', 'tan4', 
                  'mediumorchid1',  'yellowgreen', 'lightsalmon', 'tan3', "tan1", 'wheat4', 
                  '#DDAD4B','chartreuse', 'moccasin', 'mediumvioletred', 'seagreen','cadetblue1',
                  "darkolivegreen1" , "tomato3" , "#7CE3D8", 'tan2',"#F39B7FFF","#91D1C2FF",
                  'red2', 'darkblue')
  if (strsplit(version,'/')[[1]][1]=='covid'){
    cell_topic_select$initial_clustering=cell_topic_raw$initial_clustering[sub_ind]
    cell_topic_select$Status_on_day_collection_summary=cell_topic_raw$Status_on_day_collection_summary[sub_ind]
    cell_topic_select$Status=cell_topic_raw$Status[sub_ind]
    
    ## color
    col_type=manualcolors[1:18]#manual checked number
    names(col_type)=unique(cell_topic_select$initial_clustering)
    
    col_severity=manualcolors[19:24]
    names(col_severity)=unique(cell_topic_select$Status_on_day_collection_summary)

    col_status=manualcolors[25:26]
    names(col_status)=unique(cell_topic_select$Status)
    
    coll = list(cellType = col_type,status = col_status,severity=col_severity)
    plot_cell_topic=cell_topic_select[order(cell_topic_select$initial_clustering),]
    
  } else if (strsplit(version,'/')[[1]][1]=='rna_atac'){
    #nips atac
    cell_topic_select$cellType=cell_info$cell_type[sub_ind]
    col_type=manualcolors[1:22]
    names(col_type)=unique(cell_info$cell_type)
    coll=list(cellType = col_type)
    plot_cell_topic=cell_topic_select[order(cell_topic_select$cellType),]
  
  } else{#nips adt
    ## broad cell type
    cell_type0=read.csv('../useful_file/cell_type_nips_color.csv',header=T)
    cell_info_new=c()
    for (i in cell_info$cell_type){
      new=cell_type0[which(cell_type0$cellType2==i),]$cellType1
      cell_info_new=c(cell_info_new,new)
    }
    
    cell_topic_select$cellType1=cell_info_new[sub_ind]
    cell_topic_select$cellType2=cell_info$cell_type[sub_ind]
    cell_topic_select_order=cell_topic_select[with(cell_topic_select,order(cellType1,cellType2)),]
    
    #color
    col_type1=cell_type0$color1
    names(col_type1)=cell_type0$cellType1
    col_type2=cell_type0$color2
    names(col_type2)=cell_type0$cellType2
    
    coll = list(cellType1 = col_type1, cellType2 = col_type2)
    plot_cell_topic=cell_topic_select[order(cell_topic_select$cellType1),]
  }
  
  save_name=paste0('../plot/',version,'/sample_topic_cutoff',cutoff,'.png')
  plot_sample_by_topic(plot_cell_topic,version,coll,num_select_topic,save_name,is_cluster=T)
  
  ##plot legend
  if (strsplit(version,'/')[[1]][1] !='rna_adt'){#not nips adt (2 cell type annotation)
    for (i in 1:length(coll)){
      labels=names(coll[[i]])
      col_fill=coll[[i]]
      lgd=Legend(labels=labels,legend_gp=gpar(fill =col_fill))
      png(paste0('../plot/',version,'/',i,'legend.png'),
          height=2000,width=1500,res=500)
      draw(lgd)
      dev.off()
    }
  } else{#nips adt
      u=unique(cell_type0$cellType1)
      u_col=unique(cell_type0$color1)
      for (i in 1:length(u)){
        cell_type0_sub=cell_type0[cell_type0$cellType1==u[i],]
      
        labels=c(u[i],unique(cell_type0_sub$cellType2))
        col_fill=c(u_col[i],unique(cell_type0_sub$color2))
      
        lgd=Legend(labels=labels,legend_gp=gpar(fill =col_fill))
        png(paste0('../plot/',version,'/',i,'legend.png'),
            height=1000,width=1500,res=500)
        draw(lgd)
        dev.off()
     }
    }
}

####### test differentially expressed topics
if (is_diff_topic==T){
  if (version=='covid'){
  ####### differentially expressed topics, remove first column which is the topic index
    ### test cell types
    topic_initial_list=diff_topic('initial_clustering',cell_info,cell_topic[,-1],num_topic)
    topic_full_list=diff_topic('full_clustering',cell_info,cell_topic[,-1],num_topic)
    ### test COVID
    topic_covid_list=diff_topic('Status',cell_info,cell_topic[,-1],num_topic)
    ### test severity
    topic_severity_list=diff_topic('Status_on_day_collection_summary',cell_info,cell_topic[,-1],num_topic)
    ### test confounder
    topic_sex_list=diff_topic('Sex',cell_info,cell_topic[,-1],num_topic,alter='two.sided')
    topic_age_list=diff_topic('Age_interval',cell_info,cell_topic[,-1],num_topic,alter='two.sided')
    topic_smoker_list=diff_topic('Smoker',cell_info,cell_topic[,-1],num_topic,alter='two.sided')
    
    ### plot differentially expressed topics
    plot_diff_topic(topic_initial_list,width=1000,topic_select=topic_select,save_name=paste0('../plot/',version,'/diff_topic_initial.png'))
    plot_diff_topic(topic_full_list,width=1000,topic_select=topic_select,save_name=paste0('.../plot/',version,'/diff_topic_full.png'))
    
    plot_diff_topic(topic_covid_list,width=200,topic_select=topic_select,paste0('../plot/',version,'/diff_topic_covid.png'))
    plot_diff_topic(topic_severity_list,width=500,topic_select=topic_select,paste0('../plot/',version,'/diff_topic_severity_sameRange.png'))
    ## confounder
    plot_diff_topic(topic_sex_list,width=250,topic_select=topic_select,paste0('../plot/',version,'/diff_topic_sex_sameRange.png'))
    plot_diff_topic(topic_age_list,width=500,topic_select=topic_select,paste0('../plot/',version,'/diff_topic_age_sameRange.png'))
    plot_diff_topic(topic_smoker_list,width=300,topic_select=topic_select,paste0('../plot/',version,'/diff_topic_smoker_sameRange.png'))
    # 
  } else{#nips, only check cell type1
    topic_cell_list=diff_topic('cell_type',cell_info,cell_topic[,-1],num_topic)
    write.csv(topic_cell_list$topic_label_p,file=paste0('..s/plot/',version,'/topic_cell_p.csv'),quote=F,row.names=F)
    plot_diff_topic(topic_cell_list,width=1200,topic_select=topic_select,save_name=paste0('../plot/',version,'/diff_topic_cell.png'))
  }
  
}


