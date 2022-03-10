import pdb 
import os.path as osp
import pickle
from tqdm import tqdm

# conda install -c conda-forge biopython
# pip install biopython
from Bio import SeqIO


'''
This script loads hg38 and hg19 human genome assemblies as well as
mm10 mouse genome assembly, if they are available. You should run
download_[genome].sh scripts before this script.

'''


# check if data files are available
def check_available(genome_type, chr_filenames):
    """ 
    This function checks if the files for the specified genome is
        ready for preprocessing
    :@param genome_type: One of "hg38", "hg19" or "mm10" 
    :@param chr_filenames: A list of file names for the chromosome that 
        should have been downloaded
    """
    data_path_prefix = f"../data/{genome_type}"
    assert osp.exists(data_path_prefix),\
        f"Please run download_{genome_type}.sh script first."

    for filename in chr_filenames:
        assert osp.exists(osp.join(data_path_prefix, filename)),\
            f"Please run download_{genome_type}.sh script first."

def load_genome_to_dict(genome_type, chr_filenames):
    """
    Load the specified genome to a dictionary where keys are chromosome
    names and values are genomic sequence for that chromosome
    :@param genome_type: One of "hg38", "hg19" or "mm10" 
    :@param chr_filenames: A list of file names for the chromosome that 
        should have been downloaded
    """

    check_available(genome_type, chr_filenames)

    data_path_prefix = f"../data/{genome_type}"
    chr_filenames = [osp.join(data_path_prefix, chrom)\
        for chrom in chr_filenames]

    # load fasta files per chromosome
    sequences = {}
    for filename in tqdm(chr_filenames, desc="fasta files"):
        file = open(filename)
        fasta = SeqIO.parse(file, "fasta")
        for record in tqdm(fasta, leave=False, desc="records in fasta"):
            name, sequence = record.id, str(record.seq)
            print(name, f"sequence length: {len(sequence)}")
            sequences[name] = sequence
        file.close()
    
    return sequences

if __name__ == "__main__":

    # humans: chr1-22 + chrX + chrY
    chr_filenames = [f"chr{id}.fa"\
        for id in list(range(1,23)) + ["X", "Y"]]

    # load, process and save hg19
    genome_type = "hg19"
    genome = load_genome_to_dict(genome_type, chr_filenames)
    with open(f"../data/{genome_type}/{genome_type}.pkl", "wb") as f:
        pickle.dump(genome, f)

    # load, process and save hg38
    genome_type = "hg38"
    genome = load_genome_to_dict(genome_type, chr_filenames)
    with open(f"../data/{genome_type}/{genome_type}.pkl", "wb") as f:
        pickle.dump(genome, f)

    # mouse: chr1-19 + chrX + chrY
    chr_filenames = [f"chr{id}.fa"\
        for id in list(range(1,20)) + ["X", "Y"]]

    # load, process and save mm10
    genome_type = "mm10"
    genome = load_genome_to_dict(genome_type, chr_filenames)
    with open(f"../data/{genome_type}/{genome_type}.pkl", "wb") as f:
        pickle.dump(genome, f)