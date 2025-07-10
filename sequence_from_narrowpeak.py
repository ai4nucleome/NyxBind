import os
import random
from Bio import SeqIO
from collections import defaultdict
import bisect
import csv
import pdb

input_folder = 'path/to/your/input/folder'  # Replace with actual input folder path
save_folder = 'path/to/your/output/folder'  # Replace with actual output folder path
ref = 'path/to/hg19.fasta'  # Replace with actual reference genome path

# Initialize dictionaries to store chromosome sequences and genome ranges
chromosome_sequences = {}
genome_ranges = {}

def load_reference_genome(path):
    # Parse FASTA file at the given path
    for record in SeqIO.parse(path, "fasta"):
        chromosome_name = record.id
        print(chromosome_name)
        sequence = str(record.seq)
        chromosome_sequences[chromosome_name] = sequence
        genome_ranges[chromosome_name] = len(sequence)
    return chromosome_sequences, genome_ranges

# Load sequence information
chromosome_sequences, genome_ranges = load_reference_genome(ref)

# Retrieve TFBS location from BED file
def get_tfbs_location(path):
    allowed_chromosomes = {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY'}
    tfbs_by_chr = defaultdict(list)
    chr_tfbs_counts = {}

    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                chr_name = parts[0]
                if chr_name not in allowed_chromosomes:
                    continue
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                tfbs_by_chr[chr_name].append((start, end))

        sorted_tfbs_by_chr = {}
        for chr_name, intervals in tfbs_by_chr.items():
            sorted_intervals = sorted(intervals)
            sorted_tfbs_by_chr[chr_name] = sorted_intervals
            chr_tfbs_counts[chr_name] = len(sorted_intervals)

        tfbs_by_chr = sorted_tfbs_by_chr

    except FileNotFoundError:
        print(f"File {path} not found.")
        return
    except OSError as e:
        print(f"An OS error occurred: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    return tfbs_by_chr, chr_tfbs_counts

# Compute GC percentage
def gc_content(dna_sequence):
    g_count = dna_sequence.count('G')
    c_count = dna_sequence.count('C')
    total_bases = len(dna_sequence)
    if total_bases == 0:
        return 0.0
    gc_percentage = int(((g_count + c_count) / total_bases) * 100)
    return gc_percentage

# Generate non-TFBS regions by avoiding overlap
def generate_non_tfbs(tfbs_by_chr, genome_ranges, chr_tfbs_counts):
    non_tfbs_sequences = {}

    def is_overlap(tfbs_intervals, search_start, search_end):
        i = bisect.bisect_left(tfbs_intervals, (search_start, search_start))
        if i < len(tfbs_intervals) and tfbs_intervals[i][0] <= search_end:
            return True
        if i > 0 and tfbs_intervals[i - 1][1] >= search_start:
            return True
        return False

    print('Start extracting non-TFBS regions')
    for chr_name, counts in chr_tfbs_counts.items():
        chr_length = genome_ranges[chr_name]
        non_tfbs_sequences[chr_name] = []
        max_required_counts = counts * 10

        print(f'{chr_name} length: {chr_length}, tfbs count: {counts}')

        while len(non_tfbs_sequences[chr_name]) < max_required_counts:
            start = random.randint(0, chr_length - 100)
            end = start + 101

            overlap = False
            if chr_name in tfbs_by_chr:
                tfbs_intervals = sorted(tfbs_by_chr[chr_name])
                overlap = is_overlap(tfbs_intervals, start, end)

            if not overlap:
                non_tfbs_sequences[chr_name].append((start, end))

        print(f'non-TFBS count for {chr_name}: {len(non_tfbs_sequences[chr_name])}')

    return non_tfbs_sequences

# Generate TFBS sequences centered around midpoint
def generate_tfbs(tfbs_by_chr):
    tfbs_sequences = {}
    print("Start generating TFBS index")
    for chr_name, lines in tfbs_by_chr.items():
        tfbs_sequences[chr_name] = []
        for line in lines:
            start = line[0]
            end = line[1]
            middle = (start + end) // 2
            tfbs_sequences[chr_name].append((middle - 50, middle + 51))
    print("Completed generating TFBS index")
    return tfbs_sequences

# Extract sequence pairs with GC-content matching
def get_pairs(tfbs_sequences, non_tfbs_sequences):
    data = {}
    tfbs_seqs = defaultdict(list)
    non_tfbs_seqs = defaultdict(list)
    print('Start extracting TFBS sequences')

    for chr_name, seq_indices in tfbs_sequences.items():
        chrom_seq = chromosome_sequences[chr_name]
        for seq_index in seq_indices:
            extracted_seq = chrom_seq[seq_index[0]:seq_index[1]].upper()
            gc_percentage = gc_content(extracted_seq)
            if all(nucleotide in 'ATCG' for nucleotide in extracted_seq):
                tfbs_seqs[chr_name].append((extracted_seq, gc_percentage))
        print(f'{chr_name} has {len(tfbs_seqs[chr_name])} TFBS sequences')

    for chr_name, seq_indices in non_tfbs_sequences.items():
        chrom_seq = chromosome_sequences[chr_name]
        for seq_index in seq_indices:
            extracted_seq = chrom_seq[seq_index[0]:seq_index[1]].upper()
            gc_percentage = gc_content(extracted_seq)
            if all(nucleotide in 'ATCG' for nucleotide in extracted_seq):
                non_tfbs_seqs[chr_name].append((extracted_seq, gc_percentage))
        print(f'{chr_name} has {len(non_tfbs_seqs[chr_name])} non-TFBS sequences')

    print("Start pairing TFBS and non-TFBS sequences")

    seen_sequences = set()

    for chr_name, tfbs_seq in tfbs_seqs.items():
        gc_dict = {}
        for non_tfbs in non_tfbs_seqs[chr_name]:
            non_tfbs_gc = non_tfbs[1]
            non_tfbs_seq = non_tfbs[0]
            if non_tfbs_gc not in gc_dict:
                gc_dict[non_tfbs_gc] = []
            gc_dict[non_tfbs_gc].append(non_tfbs_seq)

        for tfbs in tfbs_seq:
            tfbs_gc = tfbs[1]
            tfbs_seq = tfbs[0]

            if tfbs_gc in gc_dict and gc_dict[tfbs_gc]:
                selected_non_tfbs = random.choice(gc_dict[tfbs_gc])
                if (tfbs_seq, 1) not in seen_sequences:
                    if chr_name not in data:
                        data[chr_name] = []
                    data[chr_name].append((tfbs_seq, 1))
                    seen_sequences.add((tfbs_seq, 1))

                if (selected_non_tfbs, 0) not in seen_sequences:
                    data[chr_name].append((selected_non_tfbs, 0))
                    seen_sequences.add((selected_non_tfbs, 0))

    print("Completed pairing TFBS and non-TFBS sequences")
    return data

# Save the dataset to CSV
def save_files(csv_file, data):
    file_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0

    with open(csv_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['sequence', 'label'])
        for sequence, label in data:
            writer.writerow([sequence, label])

# Main processing function for each file
def process_file(filename, input_folder, genome_ranges, save_folder):
    tfbs_file_path = os.path.join(input_folder, filename)
    tfbs_by_chr, chr_tfbs_counts = get_tfbs_location(tfbs_file_path)
    name, extension = os.path.splitext(filename)
    folder = os.path.join(save_folder, name)
    os.makedirs(folder, exist_ok=True)

    print(f'Saving file: {os.path.join(folder, "train.csv")}')

    tfbs_sequences = generate_tfbs(tfbs_by_chr)
    non_tfbs_sequences = generate_non_tfbs(tfbs_by_chr, genome_ranges, chr_tfbs_counts)
    data = get_pairs(tfbs_sequences, non_tfbs_sequences)

    train = []
    test = []
    dev = []
    for chr_name, seqs in data.items():
        if chr_name in ['chr11', 'chr12']:
            for seq in seqs:
                signal = random.choice([0, 1])
                if signal == 0:
                    dev.append(seq)
                else:
                    test.append(seq)
        else:
            for seq in seqs:
                train.append(seq)

    save_files(os.path.join(folder, 'train.csv'), train)
    save_files(os.path.join(folder, 'dev.csv'), dev)
    save_files(os.path.join(folder, 'test.csv'), test)

# Loop through all .bed files in input folder
for filename in os.listdir(input_folder):
    print(f'Processing file {filename}')
    name, extension = os.path.splitext(filename)
    if extension == '.bed': 
        process_file(filename, input_folder, genome_ranges, save_folder)
    print(f'Finished processing file {filename}')
