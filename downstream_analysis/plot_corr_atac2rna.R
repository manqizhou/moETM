#ATAC2RNA
name='ATAC2RNA'
library('R.matlab')
library(circlize)
obs = read.csv(paste0('./',name,'/nips_atac_rna_obs.csv'))## the obs of the input anndata, contains cell type for each sample
recon=readMat(paste0('./',name,'/gex_recon.mat'))
recon=recon[[1]]
original=readMat(paste0('./',name,'/gex_original.mat'))
original=original[[1]]

cell_type=unique(obs$cell_type)#cell type for each samplee
num_cell_type = length(cell_type)

original_by_cell_type = c()
recon_by_cell_type = c()
for (i in 1:num_cell_type){
  index = which(obs$cell_type==cell_type[i])
  original_by_cell_type = rbind(original_by_cell_type,colMeans(original[index, ]))
  recon_by_cell_type = rbind(recon_by_cell_type,colMeans(recon[index, ]))
}

###### plot heatmap
library(ComplexHeatmap)
png(file=paste0('./',name,'/recon_heatmap.png'),width = 450, height = 600,units='px',bg = "transparent",res=100)
col_fun = colorRamp2(c(min(recon_by_cell_type), max(recon_by_cell_type)), c("blue", "red"))
recon_h=Heatmap(recon_by_cell_type, col=col_fun,              
                cluster_columns = T,cluster_rows = T)
recon_h=draw(recon_h)
roworder=row_order(recon_h)
colorder=column_order(recon_h)
dev.off()

original_by_cell_type_order=original_by_cell_type[roworder,colorder]
png(file=paste0('./',name,'/original_heatmap.png'),width = 450, height = 600,units='px',bg = "transparent",res=100)
col_fun = colorRamp2(c(min(original_by_cell_type_order), max(original_by_cell_type_order)), c("blue", "red"))
orig_h=Heatmap(original_by_cell_type_order,col=col_fun,
               cluster_columns = F,cluster_rows = F)
draw(orig_h)
dev.off()


###scatter plot
library(ggplot2)
# x=c(original)
# y=c(recon)
x=c(original_by_cell_type)
y=c(recon_by_cell_type)

p=ggplot()+
  geom_point(aes(x=x,y=y))+
  theme_classic()+
  ylab('reconstruct')+xlab('original')+ggtitle(name)+
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 20))+
  geom_abline(intercept = 0, slope = 1, color="blue", linetype="dashed",size=2)
ggsave(p,width=9,height=7,file=paste0('./',name,'/scatter.png'))





