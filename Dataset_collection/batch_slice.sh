#!/bin/bash
# USAGE: bash ./batch_generate_slices.sh [normal/inter]
echo "Start generate Asm codes in batch!"
path=../Dataset/Raw_Dataset
result=../Result

for dir in ${path}/*
do
	# echo "Execute embedding generation for: $dir"
	for second_dir in ${dir}/*
	do
		# echo "Execute embedding generation for: $second_dir"
		if [[ "$1" == "normal" ]]; then
			echo "python ./generate_slice.py --data_path ${second_dir} --mode asm --policy normal" >> ${result}/sli_data_$1_path.temp
		elif [[ "$1" == "inter" ]]; then
			echo "python ./generate_slice.py --data_path ${second_dir} --mode asm --policy inter" >> ${result}/sli_data_$1_path.temp
		else
			echo "Please input the expected data format [normal/inter]."
		fi
	done
done

parallel -j 15 "{}" :::: ${result}/sli_data_$1_path.temp

rm -f ${result}/sli_data_$1_path.temp
