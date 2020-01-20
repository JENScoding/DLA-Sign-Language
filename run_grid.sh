#!/bin/bash
cd "${0%/*}"

for drop in {0.7, 0.6, 0.5, 1}
do   
	for mixl1l2 in {1e-6, 1e-8, 1e-10, 0}
	do
		for lambda in {0.01, 0.001, 0.0001, 0}	
		do python parameter_test.py --mixl1l2 "$mixl1l2" --lambda "$lambda" --epochs 50 --keep_prob "$drop"
			echo "Lambda is: $lambda"
		done
		echo "The mixing ratio is: $mixl1l2"
	done
	echo "Dropout is: $drop"
done


