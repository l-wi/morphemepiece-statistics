from datasets import load_dataset
from multiprocessing import Pool
import uuid
import itertools
import subprocess
import os
from transformers import BertTokenizer

OUTPUT_PATH = "./wikien"


def tokenize_morpheme(text,title, output_path):
    bcmd = "Rscript tokenize-text.R "

    process = subprocess.Popen(bcmd,stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

    output, error = process.communicate(text.encode("utf-8"))
    
    err = error.decode("utf-8")
    out = output.decode("utf-8")

    if err.strip():
        print("Skipping %s due to %s " % (title, err))

    with open(output_path, "w") as outpf:
        outpf.write(out)
        print("tokenized morph %s" % title)


def tokenize_bert(text, title, output_path):
    tz = BertTokenizer.from_pretrained("bert-base-uncased")
    out = " ".join(tz.tokenize(text))

    with open(output_path, "w") as outpf:
        outpf.write(out)
        print("tokenized bert %s" % title)

def tokenize_ds_file(json_data):
    filename = uuid.uuid4().hex 
    m_output_path = OUTPUT_PATH + "/" + filename + ".morph.txt"
    b_output_path = OUTPUT_PATH + "/" + filename + ".bert.txt"

    text = json_data['text'] + "\n"

    tokenize_morpheme(text,json_data['title'], m_output_path)
    tokenize_bert(text,json_data['title'], b_output_path)
    
def main():
    assert len(os.listdir(OUTPUT_PATH)) == 0, "Please specify an empty output folder"

    ds = load_dataset('wikipedia', "20200501.en")
    it = itertools.islice(ds['train'],100*1000)

    with Pool(16) as p:
        r =p.map(tokenize_ds_file,it)
        print("%s files processed." % len([x for x in r]))


if __name__ == '__main__':
    main()

