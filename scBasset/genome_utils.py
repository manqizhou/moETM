import pdb
import random
import numpy as np

import os.path as osp
from Bio import SeqIO


def check_available(genome_dir, chr_filenames):
    """ 
    This function checks if the files for the specified genome is
        ready for preprocessing
    :@param genome_path: File path to genome
    :@param chr_filenames: A list of file names for the chromosome that 
        should have been downloaded
    """
    assert osp.exists(genome_dir),\
        f"Please run bash script for downloading the requested genome first."

    for filename in chr_filenames:
        assert osp.exists(osp.join(genome_dir, filename)),\
            f"Please run bash script for downloading the requested genome first."

def load_genome_to_dict(genome_dir, chr_filenames):
    """
    Load the specified genome to a dictionary where keys are chromosome
    names and values are genomic sequence for that chromosome
    :@param genome_path: File path to genome
    :@param chr_filenames: A list of file names for the chromosome that 
        should have been downloaded
    """

    check_available(genome_dir, chr_filenames)

    chr_filenames = [osp.join(genome_dir, chrom)\
        for chrom in chr_filenames]

    # load fasta files per chromosome
    sequences = {}
    for filename in chr_filenames:
        file = open(filename)
        fasta = SeqIO.parse(file, "fasta")
        for record in fasta:
            name, sequence = record.id, str(record.seq)
            sequences[name] = sequence
        file.close()
    
    return sequences