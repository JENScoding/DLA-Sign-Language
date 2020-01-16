#!/bin/bash
cd "${0%/*}"
for l1 in {0,1e-3,1e-5,1e-7,1e-8,1e-10,1e-12}
do
	echo "The following runs were tested with l1 = $l1"
	for l2 in {0,1e-3,1e-5,1e-7,1e-8,1e-10,1e-12}	
	do python keep_it_lower.py --lambda1 "$l1" --lambda2 "$l2" --epochs 30 --keep_prob 1
		echo "Second parameter was setted to l2 = $l2"
	done
done


