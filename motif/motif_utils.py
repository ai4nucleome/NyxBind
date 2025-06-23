#### ::: utils for DNABERT-viz motif search ::: ####

import os
import pandas as pd
import numpy as np
from Bio.motifs import write as write_motifs  
import pandas as pd 
import numpy as np
from scipy.stats import hypergeom
import statsmodels.stats.multitest as multi
from tqdm import tqdm 
import ahocorasick 
from operator import itemgetter

def contiguous_regions(condition, len_thres=5):
    """
    Modified from and credit to: https://stackoverflow.com/a/4495197/3751373
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    
    Arguments:
    condition -- custom conditions to filter/select high attention 
            (list of boolean arrays)
    
    Keyword arguments:
    len_thres -- int, specified minimum length threshold for contiguous region 
        (default 5)
    
    Returns:
    idx -- Index of contiguous regions in sequence
    """
    
    # Find the indices where "condition" changes
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start after the change in "condition", so shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of the condition is True, prepend a 0.
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of the condition is True, append the length of the array.
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns.
    idx.shape = (-1,2)
    
    # Eliminate regions that do not satisfy the length threshold.
    idx = idx[np.argwhere((idx[:,1]-idx[:,0])>=len_thres).flatten()]
    return idx

def find_high_attention(score, min_len=5, **kwargs):
    """
    Given an array of attention scores as input, finds indices of contiguous high-attention 
    sub-regions with length greater than min_len.
    
    Arguments:
    score -- numpy array of attention scores for a sequence

    Keyword arguments:
    min_len -- int, specified minimum length threshold for contiguous region 
        (default 5)
    **kwargs -- other input arguments:
        cond -- custom conditions to filter/select high attention 
            (list of boolean arrays)
    
    Returns:
    motif_regions -- indices of high-attention regions in the sequence.
    """
    
    cond1 = (score > np.mean(score))
    cond2 = (score > 10 * np.min(score))
    cond = [cond1, cond2]
    
    cond = list(map(all, zip(*cond)))
    
    if 'cond' in kwargs:  # if custom conditions are provided, use them.
        cond = kwargs['cond']
        if any(isinstance(x, list) for x in cond):  # if the input contains multiple conditions.
            cond = list(map(all, zip(*cond)))
    
    cond = np.asarray(cond)
        
    # Find important contiguous regions with high attention.
    motif_regions = contiguous_regions(cond, min_len)
    
    return motif_regions

def count_motif_instances(seqs, motifs, allow_multi_match=False):
    """
    Use the Aho-Corasick algorithm for efficient multi-pattern matching between 
    input sequences and motif patterns to obtain counts of motif instances.
    
    Arguments:
    seqs -- list, numpy array or pandas series of DNA sequences.
    motifs -- list, numpy array or pandas series, a collection of motif patterns
              to be matched against the sequences.
    
    Keyword arguments:
    allow_multi_match -- bool, whether to allow counting multiple matches (default False).
    
    Returns:
    motif_count -- a dictionary of motif counts (dict: {motif_string: count})
    """
    
    motif_count = {}    
    A = ahocorasick.Automaton()
    for idx, key in enumerate(motifs):
        A.add_word(key, (idx, key))
        # Initialize motif count
        motif_count[key] = 0
    A.make_automaton()
    
    for seq in seqs:
        # Process each sequence
        # print(f"🔄Processing sequence: {seq}")  # Uncomment for debugging
        matches = sorted(map(itemgetter(1), A.iter(seq)))
        
        if allow_multi_match:
            for match in matches:
                match_seq = match[1]
                # Optionally assert that the key exists in the dictionary
                motif_count[match_seq] += 1
        else:  # For a particular sequence, count only once even if there are multiple matches.
            matched_seqs_in_current_seq = set()  # Use a set for efficiency.
            for match in matches:
                match_seq = match[1]
                # Optionally assert that the key exists in the dictionary.
                if match_seq not in matched_seqs_in_current_seq:
                    motif_count[match_seq] += 1
                    matched_seqs_in_current_seq.add(match_seq)
    
    return motif_count

# --- Optimized motifs_hypergeom_test function with progress bar ---
def motifs_hypergeom_test(pos_seqs, neg_seqs, motifs, p_adjust='fdr_bh', alpha=0.05, verbose=False, 
                          allow_multi_match=False, **kwargs):
    """
    Perform a hypergeometric test to determine significantly enriched motifs in positive sequences.
    Returns a list of adjusted p-values.
    
    Arguments:
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences.
    neg_seqs -- list, numpy array or pandas series of negative DNA sequences.
    motifs -- list, numpy array or pandas series, a collection of motif patterns
              to be matched against sequences.
    
    Keyword arguments:
    p_adjust -- the method used to correct for multiple testing. Options are consistent with
                statsmodels.stats.multitest (default 'fdr_bh').
    alpha -- the significance cutoff (default 0.05).
    verbose -- bool, controls verbosity (default False).
    allow_multi_match -- bool, whether to allow counting of multiple matches (default False).
    
    Returns:
    pvals -- a list of p-values.
    """
    pvals = []
    N = len(pos_seqs) + len(neg_seqs)
    K = len(pos_seqs)
    
    # If the counting step is very time-consuming, tqdm can be added.
    # The main bottleneck is usually in the subsequent loop.
    if verbose:
        print("Counting motifs in positive sequences...")
    motif_count_pos = count_motif_instances(pos_seqs, motifs, allow_multi_match=allow_multi_match)
    
    if verbose:
        print("Counting motifs in negative sequences...")
    motif_count_neg = count_motif_instances(neg_seqs, motifs, allow_multi_match=allow_multi_match)
    
    motif_count_all = {}
    # This loop is generally fast, unless the motifs list is extremely large and dictionary operations become a bottleneck.
    for motif in motifs:
        motif_count_all[motif] = motif_count_pos.get(motif, 0) + motif_count_neg.get(motif, 0)
    
    if verbose and not kwargs.get('no_hypergeom_progressbar', False):
        print("Performing hypergeometric tests (with progress bar)...")
    
    # Add a tqdm progress bar to the hypergeometric test loop.
    # Even if verbose is False, the user might want to see a progress bar.
    # Here we assume that if there are many motifs or verbose is True, the progress bar should be shown.
    if verbose:
        print("Performing hypergeometric tests...")
        iterator_motifs = tqdm(motifs, desc="Hypergeometric Test", unit="motif", leave=False, colour='green')
    else:
        iterator_motifs = tqdm(motifs, desc="Hypergeometric Test", unit="motif", leave=False, disable=kwargs.get('disable_progressbar', False), colour='green')
    
    for motif in iterator_motifs:
        n = motif_count_all.get(motif, 0)
        x = motif_count_pos.get(motif, 0)
        
        if n > 0: 
            pval = hypergeom.sf(x - 1, N, K, n)
        else: 
            pval = 1.0
        
        # The verbose printing below can be removed or adjusted since the progress bar shows progress.
        # if verbose:
        #     if pval < 1e-5 or motif_count_pos.get(motif, 0) > 0:
        #         print("Motif {}: N={}, K={}, n={}, x={}, p-value={}".format(motif, N, K, n, x, pval))
        
        pvals.append(pval)
    
    if verbose:
        print("Hypergeometric tests completed.")
    
    if p_adjust is not None and pvals:
        if verbose:
            print(f"Adjusting p-values using {p_adjust} method...")
        reject, pvals_corrected, _, _ = multi.multipletests(pvals, alpha=alpha, method=p_adjust)
        pvals = list(pvals_corrected)
        if verbose:
            print("P-value adjustment completed.")
            
    return pvals

def filter_motifs(pos_seqs, neg_seqs, motifs, cutoff=0.05, return_idx=False, **kwargs):
    """
    Wrapper function for returning the actual motifs that pass the hypergeometric test.
    
    Arguments:
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences.
    neg_seqs -- list, numpy array or pandas series of negative DNA sequences.
    motifs -- list, numpy array or pandas series, a collection of motif patterns
              to be matched against sequences.
    
    Keyword arguments:
    cutoff -- cutoff FDR/p-value to declare statistical significance (default 0.05).
    return_idx -- whether to return only the indices of the motifs (default False).
    **kwargs -- additional input arguments.
    
    Returns:
    A list of filtered motifs (or indices of the motifs).
    """
    pvals = motifs_hypergeom_test(pos_seqs, neg_seqs, motifs, **kwargs)
    if return_idx:
        return [i for i, pval in enumerate(pvals) if pval < cutoff]
    else:
        return [motifs[i] for i, pval in enumerate(pvals) if pval < cutoff]

def merge_motifs(motif_seqs, min_len=5, align_all_ties=True, **kwargs):
    """
    Function to merge similar motifs in the input motif_seqs.
    
    First, sort the keys of input motif_seqs by length. For each query motif (with length 
    guaranteed to be >= a key motif), perform pairwise alignment between them.
    
    If they can be aligned, find the best alignment among all combinations, then adjust the 
    start and end positions of the high-attention region based on left/right offsets calculated 
    from the alignment of the query and key motifs.
    
    If the query motif cannot be aligned with any existing key motifs, add it as a new key motif.
    
    Returns:
    merged_motif_seqs -- a new nested dictionary containing merged motifs.
    
    Arguments:
    motif_seqs -- nested dict with the following structure: 
        {motif: {seq_idx: idx, atten_region_pos: (start, end)}}
        where seq_idx indicates indices of pos_seqs containing the motif, and
        atten_region_pos indicates the location of the high-attention region.
    
    Keyword arguments:
    min_len -- int, specified minimum length threshold for contiguous region (default 5).
    
    align_all_ties -- bool, whether to keep all best alignments when ties are encountered (default True).
    
    **kwargs -- additional input arguments, which may include:
        - cond: custom condition used to determine successful alignment.
                Default is score > max(min_len - 1, 0.5 * (min(length(query), length(key)))).
    
    Returns:
    merged_motif_seqs -- nested dict with the same structure as motif_seqs.
    """ 
    
    from Bio import Align
    
    ### TODO: Modify algorithm to improve efficiency later.
    aligner = Align.PairwiseAligner()
    # Prohibit internal gaps.
    aligner.internal_gap_score = -10000.0
    
    merged_motif_seqs = {}
    for motif in sorted(motif_seqs, key=len):  # Query motif.
        if not merged_motif_seqs:  # If empty, add the first motif.
            merged_motif_seqs[motif] = motif_seqs[motif]
        else:  # If not empty, compare and see if it can be merged.
            # First, compute alignment scores for all key motifs.
            alignments = []
            key_motifs = []
            for key_motif in merged_motif_seqs.keys():  # Key motifs.
                if motif != key_motif:  # Do not align the motif with itself.
                    # First is the query, second is the key in the new dict.
                    # Query is guaranteed to be of length >= key after sorting.
                    alignment = aligner.align(motif, key_motif)[0]
                    
                    # Condition to declare a successful alignment.
                    cond = max((min_len - 1), 0.5 * min(len(motif), len(key_motif)))
                    
                    if 'cond' in kwargs:
                        cond = kwargs['cond']  # Override the default if provided.
                        
                    if alignment.score >= cond:  # There exists a key that can be aligned.
                        alignments.append(alignment)
                        key_motifs.append(key_motif)
            
            if alignments:  # If an alignment exists, find the one with maximum score.
                best_score = max(alignments, key=lambda alignment: alignment.score)
                best_idx = [i for i, score in enumerate(alignments) if score == best_score]
                
                if align_all_ties:
                    for i in best_idx:
                        alignment = alignments[i]
                        key_motif = key_motifs[i]
                        
                        # Calculate left offset (query start - key start).
                        left_offset = alignment.aligned[0][0][0] - alignment.aligned[1][0][0]
                        if (alignment.aligned[0][0][1] <= len(motif)) and \
                           (alignment.aligned[1][0][1] == len(key_motif)):  # inside alignment
                            right_offset = len(motif) - alignment.aligned[0][0][1]
                        elif (alignment.aligned[0][0][1] == len(motif)) and \
                             (alignment.aligned[1][0][1] < len(key_motif)):  # left shift
                            right_offset = alignment.aligned[1][0][1] - len(key_motif)
                        elif (alignment.aligned[0][0][1] < len(motif)) and \
                             (alignment.aligned[1][0][1] == len(key_motif)):  # right shift
                            right_offset = len(motif) - alignment.aligned[0][0][1]
                        
                        # Add sequence indices from the query motif to the merged key motif.
                        merged_motif_seqs[key_motif]['seq_idx'].extend(motif_seqs[motif]['seq_idx'])
                        
                        # Adjust high-attention region positions based on the offsets.
                        new_atten_region_pos = [(pos[0] + left_offset, pos[1] - right_offset) \
                                                for pos in motif_seqs[motif]['atten_region_pos']]
                        merged_motif_seqs[key_motif]['atten_region_pos'].extend(new_atten_region_pos)
                
                else:
                    alignment = alignments[best_idx[0]]
                    key_motif = key_motifs[best_idx[0]]
                    
                    # Calculate offsets for alignment.
                    left_offset = alignment.aligned[0][0][0] - alignment.aligned[1][0][0]
                    if (alignment.aligned[0][0][1] <= len(motif)) and \
                       (alignment.aligned[1][0][1] == len(key_motif)):  # inside alignment
                        right_offset = len(motif) - alignment.aligned[0][0][1]
                    elif (alignment.aligned[0][0][1] == len(motif)) and \
                         (alignment.aligned[1][0][1] < len(key_motif)):  # left shift
                        right_offset = alignment.aligned[1][0][1] - len(key_motif)
                    elif (alignment.aligned[0][0][1] < len(motif)) and \
                         (alignment.aligned[1][0][1] == len(key_motif)):  # right shift
                        right_offset = len(motif) - alignment.aligned[0][0][1]
                    
                    merged_motif_seqs[key_motif]['seq_idx'].extend(motif_seqs[motif]['seq_idx'])
                    new_atten_region_pos = [(pos[0] + left_offset, pos[1] - right_offset) \
                                            for pos in motif_seqs[motif]['atten_region_pos']]
                    merged_motif_seqs[key_motif]['atten_region_pos'].extend(new_atten_region_pos)
            
            else:  # If no alignment can be made, add the motif as a new key.
                merged_motif_seqs[motif] = motif_seqs[motif]
    
    return merged_motif_seqs

def make_window(motif_seqs, pos_seqs, window_size=24):
    """
    Function to extract fixed, equal-length sequences centered on high-attention motif instances.
    
    Returns a new dict containing sequences with the fixed window_size.
    
    Arguments:
    motif_seqs -- nested dict with the following structure: 
        {motif: {seq_idx: idx, atten_region_pos: (start, end)}}
        where seq_idx indicates indices of pos_seqs containing the motif, and
        atten_region_pos indicates the location of the high-attention region.
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences.
    
    Keyword arguments:
    window_size -- int, specified window size for the final motif length (default 24).
    
    Returns:
    new_motif_seqs -- nested dict with the same structure as motif_seqs.
    """
    new_motif_seqs = {}
    
    # Extract fixed-length sequences based on the window_size.
    for motif, instances in motif_seqs.items():
        new_motif_seqs[motif] = {'seq_idx': [], 'atten_region_pos': [], 'seqs': []}
        for i, coord in enumerate(instances['atten_region_pos']):
            atten_len = coord[1] - coord[0]
            if (window_size - atten_len) % 2 == 0:  # even case
                offset = (window_size - atten_len) / 2
                new_coord = (int(coord[0] - offset), int(coord[1] + offset))
                if (new_coord[0] >= 0) and (new_coord[1] < len(pos_seqs[instances['seq_idx'][i]])):
                    # Append data
                    new_motif_seqs[motif]['seq_idx'].append(instances['seq_idx'][i])
                    new_motif_seqs[motif]['atten_region_pos'].append((new_coord[0], new_coord[1]))
                    new_motif_seqs[motif]['seqs'].append(pos_seqs[instances['seq_idx'][i]][new_coord[0]:new_coord[1]])
            else:  # odd case
                offset1 = (window_size - atten_len) // 2
                offset2 = (window_size - atten_len) // 2 + 1
                new_coord = (int(coord[0] - offset1), int(coord[1] + offset2))
                if (new_coord[0] >= 0) and (new_coord[1] < len(pos_seqs[instances['seq_idx'][i]])):
                    # Append data
                    new_motif_seqs[motif]['seq_idx'].append(instances['seq_idx'][i])
                    new_motif_seqs[motif]['atten_region_pos'].append((new_coord[0], new_coord[1]))
                    new_motif_seqs[motif]['seqs'].append(pos_seqs[instances['seq_idx'][i]][new_coord[0]:new_coord[1]])
    
    return new_motif_seqs

### Full pipeline for motif analysis
def motif_analysis(pos_seqs,
                   neg_seqs,
                   pos_atten_scores,
                   window_size=20,
                   min_len=4,
                   pval_cutoff=0.005,
                   min_n_motif=3,
                   top_k=1,
                   align_all_ties=True,
                   save_file_dir=None,
                   background_freqs=None,
                   pwm_pseudocount=1.0,
                   **kwargs):
    """
    Wrapper function for the full motif analysis tool based on DNABERT-viz.
    
    Arguments:
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences.
    neg_seqs -- list, numpy array or pandas series of negative DNA sequences.
    pos_atten_scores -- numpy array of attention scores for positive DNA sequences.
    
    Keyword arguments:
    window_size -- int, specified window size for the final motif length (default 24).
    min_len -- int, specified minimum length threshold for contiguous region (default 5).
    pval_cutoff -- float, cutoff FDR/p-value to declare statistical significance (default 0.005).
    min_n_motif -- int, minimum number of instances within a motif required for filtering (default 3).
    align_all_ties -- bool, whether to keep all best alignments when ties are encountered (default True).
    save_file_dir -- str, path to save output files (default None).
    **kwargs -- additional input arguments, which may include:
        - verbose: bool, controls verbosity.
        - atten_cond: custom conditions to filter/select high attention (list of boolean arrays).
        - return_idx: whether to return only the indices of the motifs.
        - align_cond: custom condition used to declare successful alignment.
                     Default is score > max(min_len - 1, 0.5 * (minimum length of two motifs)).
    
    Returns:
    merged_motif_seqs -- nested dict with the following structure: 
        {motif: {seq_idx: idx, atten_region_pos: (start, end)}}
        where seq_idx indicates indices of pos_seqs containing the motif, and
        atten_region_pos indicates the location of the high-attention region.
    """
    from Bio import motifs
    from Bio.Seq import Seq
    import math  # May be used internally by Biopython or for manual calculations.
   
    verbose = kwargs.get('verbose', False)
    
    if verbose:
        print("*** Begin motif analysis ***")
    pos_seqs = list(pos_seqs)
    neg_seqs = list(neg_seqs)
    
    if verbose:
        print("* pos_seqs: {}; neg_seqs: {}".format(len(pos_seqs), len(neg_seqs)))
    if background_freqs is None:
        background_freqs = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}  # Uniform background for DNA.
    
    assert len(pos_seqs) == len(pos_atten_scores)
    
    max_seq_len = len(max(pos_seqs, key=len))
    motif_seqs = {}
    
    ## Find high-attention motif regions.
    if verbose:
        print("* Finding high-attention motif regions")
    for i, score in enumerate(pos_atten_scores):
        seq_len = len(pos_seqs[i])
        score = score[0:seq_len]
        
        # Handle optional custom conditions
        if 'atten_cond' in kwargs:
            motif_regions = find_high_attention(score, min_len=min_len, cond=kwargs['atten_cond'])
        else:
            motif_regions = find_high_attention(score, min_len=min_len)
            
        for motif_idx in motif_regions:
            seq = pos_seqs[i][motif_idx[0]:motif_idx[1]]
            if seq not in motif_seqs:
                motif_seqs[seq] = {'seq_idx': [i], 'atten_region_pos': [(motif_idx[0], motif_idx[1])]}
            else:
                motif_seqs[seq]['seq_idx'].append(i)
                motif_seqs[seq]['atten_region_pos'].append((motif_idx[0], motif_idx[1]))
    
    # Filter motifs using hypergeometric test.
    return_idx = kwargs.pop('return_idx', False)
    if verbose:
        print("* Filtering motifs by hypergeometric test")
    motifs_to_keep = filter_motifs(pos_seqs, 
                                   neg_seqs, 
                                   list(motif_seqs.keys()), 
                                   cutoff=pval_cutoff, 
                                   return_idx=return_idx, 
                                   **kwargs)
    
    motif_seqs = {k: motif_seqs[k] for k in motifs_to_keep}
    
    # Merge similar motif instances.
    if verbose:
        print("* Merging similar motif instances")
    if 'align_cond' in kwargs:
        merged_motif_seqs = merge_motifs(motif_seqs, min_len=min_len, 
                                         align_all_ties=align_all_ties,
                                         cond=kwargs['align_cond'])
    else:
        merged_motif_seqs = merge_motifs(motif_seqs, min_len=min_len,
                                         align_all_ties=align_all_ties)
        
    # Create fixed-length window sequences.
    if verbose:
        print("* Making fixed_length window = {}".format(window_size))
    merged_motif_seqs = make_window(merged_motif_seqs, pos_seqs, window_size=window_size)
    
    # Remove motifs with fewer instances than the threshold.
    if verbose:
        print("* Removing motifs with less than {} instances".format(min_n_motif))
    merged_motif_seqs = {k: coords for k, coords in merged_motif_seqs.items() if len(coords['seq_idx']) >= min_n_motif}
    
    if save_file_dir is not None:
        if verbose:
            print("* Saving outputs to directory")
        os.makedirs(save_file_dir, exist_ok=True)
    
        # Sort motifs by frequency and select the top motifs (e.g., top_k).
        sorted_motifs = sorted(merged_motif_seqs.items(), key=lambda x: len(x[1]['seq_idx']), reverse=True)
        top_motifs = sorted_motifs[:top_k]  # Keep only the top_k motifs.
    
        # Save the top motifs.
        for motif, instances in top_motifs:
            # Save sequences to a text file.
            with open(os.path.join(save_file_dir, 'motif_{}_{}.txt'.format(motif, len(instances['seq_idx']))), 'w') as f:
                for seq in instances['seqs']:
                    f.write(seq + '\n')
    
            # Create a BioPython motif object.
            seqs_for_biopython = [Seq(v) for v in instances['seqs']]
            m = motifs.create(seqs_for_biopython)
    
            # Compute and save the PWM in MEME format.
            try:
                m.pseudocounts = pwm_pseudocount
                m.background = background_freqs
    
                # Save the PWM using Biopython's write_motifs in MEME format.
                meme_file_path = os.path.join(save_file_dir, f"motif_{motif}.jaspar")
                with open(meme_file_path, 'w') as f_meme:
                    meme_str = write_motifs([m], "jaspar")
                    f_meme.write(meme_str)
    
                if verbose:
                    print(f"  - Saved MEME-format PWM for motif '{motif}' to {meme_file_path}")
            except Exception as e:
                if verbose:
                    print(f"  - Error saving MEME PWM for motif '{motif}': {e}")
    
            # Save a WebLogo image.
            m.weblogo(os.path.join(save_file_dir, "motif_{}_{}_weblogo.png".format(motif, len(instances['seq_idx']))),
                      format='png_print',
                      show_fineprint=False,
                      show_ends=False,
                      color_scheme='color_classic')
            if verbose:
                print(f"  - Saved WebLogo for motif '{motif}'")
    
    return merged_motif_seqs


