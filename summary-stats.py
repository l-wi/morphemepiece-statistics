import os
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np

INPUT_PATH = "wikien1000"

def count_file(inp_file):
    inp_file = INPUT_PATH + "/" + inp_file    

    type = None

    if "bert.txt" in inp_file:
        type = "bert"
    elif "morph.txt" in inp_file:
        type = "morph"

    uuid = inp_file.split(".")[0]

    assert type in {"bert","morph"}
    
    with open(inp_file) as f:
        text = f.read()
        tokens = text.split(" ")

    return uuid, type, Counter(tokens)

def compute_summary_counter(data):
    agg = Counter()

    for d in data:
        agg += d[2]

    return agg


def print_top_n_total(bert,morph,most_common:int):
    print("### TOP N tokens ###")

    l1 = bert.most_common(most_common)
    l2 = morph.most_common(most_common)

    print_top_n(l1,l2)


def print_top_n(l1,l2):
    max_word_length = max(compute_longest(l1),compute_longest(l2))

    print("BERT\t\t\t\tValue\t\tMORPH\t\t\t\tValue")
    for e1,e2 in zip(l1,l2):
        print("%s\t\t\t\t%s\t\t%s\t\t\t\t%s\t\t" % (pad_word(e1[0],max_word_length),e1[1],pad_word(e2[0],max_word_length),e2[1]))


def compute_longest(counter_list):
    return max([len(x[0]) for x in counter_list])

def pad_word(word, max_len):
    while len(word) < max_len:
        word += " "
    return word


def print_number_of_unique_tokens(bert,morph):
    u_bert = vocab_size(bert) 
    u_morph = vocab_size(morph) 

    print("vocab size BERT:\t%d" % u_bert)
    print("vocab size MORPH:\t%d" % u_morph)


def vocab_size(counter):
    return len(dict(counter).keys())


def overlap(bert,morph):
    total_bert = sum(bert.values())
    total_morph = sum(morph.values())

    total = total_bert + total_morph

    d_bert = dict(bert)
    d_morph = dict(morph)

    common_keys = d_bert.keys() & d_morph.keys()
    
    sum_common = sum([d_bert[x] for x in common_keys]) + sum([d_morph[x] for x in common_keys])

    return sum_common / total


def print_vocab_overlap(bert, morph):
    k_bert = dict(bert).keys()
    k_morph = dict(morph).keys()

    common = k_bert & k_morph

    overlap =  len(common) / (len(k_bert) + len(k_morph))
    print("Overlap in vocab: %.3f" %  overlap)


def print_doc_stats(data):

    data = sorted(data, key=lambda x: x[0])

    data_group_it = itertools.groupby(data, key=lambda x: x[0])

    overlaps = []
    bert_vocab_sizes = []
    morph_vocab_sizes = []

    for _, g in data_group_it:
        entries = list(g)
        
        #TODO fix this broken groups when exporting wikipedia data
        if len(entries) != 2:
            print("skipping broken group %s" % (entries[0][0]))
            continue

        entry1 = entries[0]
        entry2 = entries[1]

        assert entry1[0] == entry2[0]

        percent_overlap = overlap(entry1[2],entry2[2])
        overlaps.append(percent_overlap)

        vocab_size_e1 = vocab_size(entry1[2])
        vocab_size_e2 = vocab_size(entry2[2])

        if entry1[1] == "morph":
            bert_vocab_sizes.append(vocab_size_e2)
            morph_vocab_sizes.append(vocab_size_e1)
        else:
            morph_vocab_sizes.append(vocab_size_e2)
            bert_vocab_sizes.append(vocab_size_e1)

    m_overlap = np.mean(overlaps) 
    sd_overlap = np.std(overlaps)
    
    m_vocab_bert = np.mean(bert_vocab_sizes)
    sd_vocab_bert = np.std(bert_vocab_sizes)

    m_vocab_morph = np.mean(morph_vocab_sizes)
    sd_vocab_morph = np.std(morph_vocab_sizes)

    print("Mean document overlap in tokens:\tM=%.3f SD=%.3f" % (m_overlap, sd_overlap ) )
    print("Mean BERT document vocab size:\t\tM=%.3f SD=%.3f" % (m_vocab_bert, sd_vocab_bert) )
    print("Mean MOPRH document vocab size:\t\tM=%.3f SD=%.3f" % (m_vocab_morph, sd_vocab_morph) )


def plot_token_histograms(bert, morph):
    plot_token_histogram([x[1] for x in bert.most_common(500)], "bert_token_top500_hist.svg")
    plot_token_histogram([x[1] for x in morph.most_common(500)], "morph_token_top500_hist.svg")


def plot_token_histogram(vals, fname):
    indexes = np.arange(len(vals))
    plt.bar(indexes,vals)
    plt.title(fname)
    plt.tight_layout()
    plt.savefig("gfx/%s" % fname)
    plt.close()


def print_top_n_distinct_tokens(bert, morph,n):
    print("### Top %d distinct tokens ###" % n)
    only_morph = compute_distinct_top_n(morph,bert,n)
    only_bert = compute_distinct_top_n(bert,morph,n)

    print_top_n(only_bert,only_morph)


def compute_distinct_top_n(retain, remove, most_common):
    d_remove = dict(remove)
    d_retain = dict(retain)

    only_retain_keys = d_retain.keys()-d_remove.keys()

    d_only_retain = { k: d_retain[k] for  k in only_retain_keys}

    counter_only_retain = Counter(d_only_retain)

    return counter_only_retain.most_common(most_common)


def print_interesting_vocab_inclusions(bert,morph):
    print("testing some intersting vocab incusions")

    k_bert = dict(bert).keys()
    k_morph = dict(morph).keys()
   
    bert_contains = ["##ed","##ing","##s"]

    for tok in bert_contains:
        print("%s in BERT:\t%s" % (tok, tok in k_bert))


def plot_relative_coverage(bert,morph, n):
    total_bert = sum(bert.values())
    total_morph = sum(morph.values())

    
    bert_most_common_vals = [x[1] for x in bert.most_common(n)]
    morph_most_common_vals = [x[1] for x in morph.most_common(n)]

    bert_partialsum = np.array(list(itertools.accumulate(bert_most_common_vals)))
    morph_partialsum = np.array(list(itertools.accumulate(morph_most_common_vals)))

    bert_relative = bert_partialsum / total_bert
    morph_relative = morph_partialsum / total_morph

    
    plot_token_histogram(bert_relative, "bert_relative_coverage_most_%d" % n)
    plot_token_histogram(morph_relative, "morph_relative_coverage_most_%d" % n)

def main():

    most_common = 20

    files = os.listdir(INPUT_PATH)

    with Pool(8) as p:
        count_map = p.map(count_file, files)

        raw_data = [data for data in count_map]

        bert_data = [d for d in raw_data if d[1] == 'bert']
        morph_data = [d for d in raw_data if d[1] == 'morph']

        bert_summary = compute_summary_counter(bert_data)
        morph_summary = compute_summary_counter(morph_data)

        print_number_of_unique_tokens(bert_summary, morph_summary)

        print_vocab_overlap(bert_summary, morph_summary) 

        print_doc_stats(raw_data)
        
        print_top_n_total(bert_summary, morph_summary, most_common)
        
        print_top_n_distinct_tokens(bert_summary, morph_summary, most_common)

        print_interesting_vocab_inclusions(bert_summary,morph_summary)

        plot_token_histograms(bert_summary, morph_summary)

        plot_relative_coverage(bert_summary,morph_summary, 2500)



if __name__ == '__main__':
    main()
