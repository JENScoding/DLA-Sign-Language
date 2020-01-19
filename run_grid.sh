#!/bin/bash
cd "${0%/*}"
for mixl1l2 in {1e-8,1e-10,0}
do
	echo "The following runs were tested with l1 = $l1"
	for lambda in {0.005,0.001,0.0005}	
	do python keep_it_lower.py --lambda1 "$l1" --lambda2 "$l2" --epochs 30 --keep_prob 1
		echo "Second parameter was setted to l2 = $l2"
	done
done


