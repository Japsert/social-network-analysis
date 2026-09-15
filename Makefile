.PHONY: tags filter

# takes a long time
tags: 1_tags.py
	python 1_tags.py data/socialbm0311.tsv -o 2_tags_desc.tsv

filter: 3_filter.sh
# head -32 is a hack -> >100
	./3_filter.sh | head -32 > 4_filtered_tags.tsv
