#!/bin/sh
# sh script to run python propagation generator

H="1"
V="1" 
F="0.5 1 1.5 2"
i=nmf              
g=../data/graph/barabasi/barabasi_edgelist.txt            
t=10              
d=10 


for h in $H;
do 
	for v in $V;
	do 
		for f in $F;
		do 
	  		time python3 womg --graph $g --virality $v --homophily $h --int_mode $i --docs $d --topics $t --infl_strength $f 
	  	done
	done
done