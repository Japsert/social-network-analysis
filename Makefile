.PHONY: tags filter

# takes a long time
tags: 1_tags.py
	python 1_tags.py data/socialbm0311.tsv -o 2_tags_desc.tsv

filter: 3_filter.sh
	./3_filter.sh > 4_filtered_tags
