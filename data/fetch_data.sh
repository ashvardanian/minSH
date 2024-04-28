# English Leipzig Corpora Collection
# 124 MB, 1'000'000 lines of ASCII, 8'388'608 tokens of mean length 5
wget --no-clobber -O leipzig1M.txt https://introcs.cs.princeton.edu/python/42sort/leipzig1m.txt 

# Hutter Prize "enwik9" dataset for compression
# 1 GB (0.3 GB compressed), 13'147'025 lines of ASCII, 67'108'864 tokens of mean length 6
wget --no-clobber -O enwik9.zip http://mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip && rm enwik9.zip && mv enwik9 enwik9.txt

# XL Sum dataset for multilingual extractive summarization
# 4.7 GB (1.7 GB compressed), 1'004'598 lines of UTF8, 268'435'456 tokens of mean length 8
wget --no-clobber -O xlsum.csv.gz https://github.com/ashvardanian/xl-sum/releases/download/v1.0.0/xlsum.csv.gz
gzip -d xlsum.csv.gz