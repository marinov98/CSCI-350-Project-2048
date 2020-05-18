#!/bin/bash
# Owen Kunhardt
# CSCI 350: 2048 Project
# Used to help create 2048 dataset

echo "Playing $1 games $2 times and storing results in $3"
echo

for ((i = 1; i <= "$2"; i++))
do
	echo "Batch $i"
	echo 
	python3 GameManager_3.py l "$3" "$1"
	echo
done
