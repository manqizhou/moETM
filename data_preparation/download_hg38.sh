#!/bin/bash


# prepare data folder for downloads
mkdir -p ../data
cd ../data
mkdir -p hg38
cd hg38

# download sequence information of chromosomes one by one 
# from https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr2.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr3.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr4.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr5.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr6.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr7.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr8.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr9.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr10.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr11.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr12.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr13.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr14.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr15.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr16.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr17.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr18.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr19.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr20.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chrX.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chrY.fa.gz

# extract the downloaded files
gunzip chr1.fa.gz
gunzip chr2.fa.gz
gunzip chr3.fa.gz
gunzip chr4.fa.gz
gunzip chr5.fa.gz
gunzip chr6.fa.gz
gunzip chr7.fa.gz
gunzip chr8.fa.gz
gunzip chr9.fa.gz
gunzip chr10.fa.gz
gunzip chr11.fa.gz
gunzip chr12.fa.gz
gunzip chr13.fa.gz
gunzip chr14.fa.gz
gunzip chr15.fa.gz
gunzip chr16.fa.gz
gunzip chr17.fa.gz
gunzip chr18.fa.gz
gunzip chr19.fa.gz
gunzip chr20.fa.gz
gunzip chr21.fa.gz
gunzip chr22.fa.gz
gunzip chrX.fa.gz
gunzip chrY.fa.gz

