###
source('./utilities.R')
num_topic=100
version='rna_atac'

#read gene by topic matrix, and all gene names
rna_topic=read.csv(paste0('../Result/',version,'/topic_rna.csv'),header=T)
rna_names=rna_topic[,1]
rna_topic=rna_topic[,-1]
rownames(rna_topic)=rna_names
colnames(rna_topic)=paste0('topic',1:num_topic)

#read protein by topic matrix, and all protein names
atac_topic=read.csv(paste0('../Result/',version,'/topic_atac.csv'),header=T)
peak_names=atac_topic[,1]
atac_topic=atac_topic[,-1]
rownames(atac_topic)=peak_names
colnames(atac_topic)=paste0('topic',1:num_topic)

##read all gene locations
gene2loci=read.csv('../useful_file/rna_atac/gene2loci2.csv',header=T)

##split peak name
peak_split=c()
for (i in peak_names){peak_split=rbind(peak_split,strsplit(i,'-')[[1]])}
peak_split=data.frame(peak_split)
colnames(peak_split)=c('chr_name','start','end')
peak_split$start=as.numeric(peak_split$start)
peak_split$end=as.numeric(peak_split$end)

##
library('GenomicRanges')
peak_gr=GRanges(seqnames = peak_split$chr_name,
                ranges = IRanges(peak_split$start, end = peak_split$end))
rna_gr=GRanges(seqnames = gene2loci$chr_name,
               ranges = IRanges(gene2loci$start_position, 
                                end = gene2loci$end_position,
                                names = gene2loci$gene_name))

simu_topic_rna=c()
for (i in 1:100){#take about 1hr
  topic=c()#converted topic_i x gene
  peak_topic_v=atac_topic[,i]#1 x num_peak
  for (j in rna_names){#gene names
    d=distance(rna_gr[j],peak_gr)
    d=ifelse(d<=150000,1,0)#distance constraint
    ind=which(d==1)#index for peak
    for (k in ind){#check non-0 peak correlation
      corr=cor(as.numeric(rna_topic[j,]),as.numeric(atac_topic[k,]))
      if (corr<=0){d[k]=0}#positive constraint
    }
    d[is.na(d)]=0
    topic=c(topic,peak_topic_v%*%d) 
  }
  simu_topic_rna=rbind(simu_topic_rna,topic)
}
write.csv(simu_topic_rna,file=paste0('../data/',version,'/simu_topic_rna.csv'),row.names = F,quote = F)

##
#imu_topic_rna=read.csv('../data/rna_atac/v2/simu_topic_rna.csv',header=T)
simu_rna_topic=t(simu_topic_rna)
rownames(simu_rna_topic)=rna_names
colnames(simu_rna_topic)=paste0('topic',1:num_topic)

### analyze rna_topic and simu_rna_topic
##correlation
plot_corr=c()
for (i in 1:num_topic){
  plot_corr=c(plot_corr,cor(rna_topic[,i],simu_rna_topic[,i]))
}
save_path=paste0('../plot/',version)
write.csv(plot_corr,file=paste0(save_path,'/plot_correlation.csv'),quote=F)

corr_color=rep(NA,100)
corr_color[plot_corr>=0]='blue'
corr_color[plot_corr<0]='red'
corr_plot=data.frame(x=1:100,y=plot_corr,color=corr_color)
save_name=paste0(save_path,'/gene_atac_corr.png')
plot_correlation(corr_plot,save_name)

## top features
gene_ids=get_feature_orders(rna_topic,num_topic=100)
simu_gene_ids=get_feature_orders(simu_rna_topic,num_topic=100)

topic_select=c(3,32,30,83,21,79,50)

top_feature_num=5
save_path=paste0('../plot/',version,'/')

plot_top_feature_in_selected_topic('atac',top_feature_num,simu_gene_ids,simu_rna_topic,topic_select,save_path,'select')
plot_top_feature_in_selected_topic('gene',top_feature_num,gene_ids,rna_topic,topic_select,save_path,'select')

## save top5 features as csv
top_rna=c()
top_atac=c()
for (i in topic_select){
  top_rna=cbind(top_rna,gene_ids[1:5,i])
  top_atac=cbind(top_atac,simu_gene_ids[1:5,i])
}


##sample x topic heatmap
version='rna_atac'
is_plot=T;is_diff_topic=F
type='rna_atac'
topic_select=','
is_all=F
args=c(num_topic,version,type,topic_select,is_plot,is_diff_topic,is_all)
system(paste(c('Rscript','./plot_sample_by_topic.R',args),collapse=' '))
#
is_plot=F;is_diff_topic=T
topic_select='30,78,3,72,84,51,55,92,50,73,39,46,33,59,12,21,79,16,98,32,25,26,38,83,88'
args=c(num_topic,version,type,topic_select,is_plot,is_diff_topic,is_all)
system(paste(c('Rscript','./plot_sample_by_topic.R',args),collapse=' '))


## prepare GSEA
############# prepare GSEA files
#rna_topic
#simu_rna_topic=t(simu_topic_rna)
save_path_g=paste0('../data/',version,'/rna/')
save_path_p=paste0('../data/',version,'/atac/')

for (i in 1:num_topic){
  value=cbind(rownames(rna_topic),rna_topic[,i])
  write.table(value,file=paste0(save_path_g,'geneName_topic',i,'.rnk'),quote=F,sep='\t',row.names = F,col.names=F)
  value=cbind(rownames(simu_rna_topic),simu_rna_topic[,i])
  write.table(value,file=paste0(save_path_p,'geneName_topic',i,'.rnk'),quote=F,sep='\t',row.names = F,col.names=F)
}

##### prepare MEME input
## top 1000 peaks
library(httr)
library(jsonlite)
library(xml2)
server <- "http://rest.ensembl.org"

###
topic_select=c(3,32,30,83,21,79,50)
#t=3
for (to in topic_select){
  peak=atac_topic[,to]
  top_peaks=peak_names[order(peak,decreasing=T)[1:100]]
  
  top_peaks_input=c()
  for (i in top_peaks){
    s=strsplit(i,'-')[[1]]#split
    t=paste0(s[1],':',s[2],'..',s[3],':1?')
    top_peaks_input=c(top_peaks_input,t)
  }
  
  #get sequences
  seqs=c()
  for (i in top_peaks_input){
    ext=paste0("/sequence/region/human/",i)
    r <- GET(paste(server, ext, sep = ""), content_type("text/x-fasta"))
    stop_for_status(r)
    seqs=c(seqs,content(r))
  }
  out_file=file(paste0("../data/rna_atac/seq/seqs100_topic",to,".fasta"))
  writeLines(seqs,out_file)
  close(out_file)
}

