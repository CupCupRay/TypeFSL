#!/bin/bash
# Usage: bash ./use_ctags.sh
echo "Start to apply ctags to analyze the source codes!"
path=../Dataset/Source_Dataset/Code
res_path=../Dataset/Source_Dataset/Index

for dir in ${path}/*
do
	echo "Use ctags for: ${dir}"
	file_name=${dir#*Dataset/Code/}
	/usr/local/bin/ctags -R -x --c-kinds=fp --extras=+q ${dir}/ > ${res_path}/${file_name}.tags
	# --c-kinds=fp
done