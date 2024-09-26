#!/bin/bash
# USAGE: bash ./batch_label.sh [agg/normal/inter]
echo "Start labels in batch!"
possible_arch=("arm-32" "arm-64" "mips-32" "mips-64" "x86-32" "x86-64")
possible_opt=("O0" "O1" "O2" "O3" "bcfobf" "cffobf" "subobf")
result=../Result

for arch in ${possible_arch[*]}
do
	for opt in ${possible_opt[*]}
	do
		echo "Execute label generation for: ${arch} ${opt} $1"
		if [[ "$1" == "agg" ]]; then
			echo "python generate_label.py --arch ${arch} --opt_level ${opt} --mode asm --policy agg" >> ${result}/label_data_$1_path.temp
		elif [[ "$1" == "inter" ]]; then
			echo "python generate_label.py --arch ${arch} --opt_level ${opt} --mode asm --policy inter" >> ${result}/label_data_$1_path.temp
		elif [[ "$1" == "normal" ]]; then
			echo "python generate_label.py --arch ${arch} --opt_level ${opt} --mode asm --policy normal" >> ${result}/label_data_$1_path.temp
		else
			echo "Please input the correct argument [asm/mlil/hlil] [agg/normal/inter]"
		fi
	done
done

parallel -j 15 "{}" :::: ${result}/label_data_$1_path.temp

rm -f ${result}/label_data_$1_path.temp
