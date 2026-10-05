.PHONY: tags filter-tags filter-rows

# takes a long time
tags: 1_tags.py
	uv run 1_tags.py data/socialbm0311.tsv -o 2_tags_desc.tsv

filter-tags: 3_filter.sh 2_tags_desc.tsv
# head -32 is a hack -> >100
	./3_filter.sh | head -32 > 4_filtered_tags.tsv

filter-rows: 6_filter_rows.py 5_manually_filtered_tags.txt
	uv run 6_filter_rows.py data/socialbm0311.tsv 5_manually_filtered_tags.txt 7_filtered_rows.tsv
