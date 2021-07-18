# Imports
from transformers import pipeline
import math
import os
import re
import sys

## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization")

# Data File Name
file_name = sys.argv[1]

from pathlib import Path
text = Path(file_name).read_text()

text_updated = re.sub(r"([0-1]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9] - ", "", text)
text_updated = re.sub(r"\n\n[a-zA-Z]+(([',. -][a-zA-Z ])?[a-zA-Z]*)*:\n", "\n ", text_updated)

def splitter(n, s):
  pieces = s.split()
  return [" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n)]

summary_text_arr = []
chunk_size = 750
for idx, chunk in enumerate(splitter(chunk_size, text_updated)):
  summary_text_arr.append(summarizer(chunk, max_length=40, min_length=10, do_sample=False)[0]['summary_text'])
  print('%d/%d' % (idx + 1, len(splitter(chunk_size, text_updated))))

print("Summary:"+".".join(summary_text_arr))
with open("{0}_Summary.{1}".format(*file_name.rsplit('.', 1)), "w") as text_file:
  text_file.write("Summary:"+".".join(summary_text_arr))
