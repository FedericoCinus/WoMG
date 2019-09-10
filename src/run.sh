#!/bin/sh
# sh script to run python propagation generator

H="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1"
V="0.5 1 1.5 2"  
i=nmf              
g=../data/graph/barabasi/barabasi_edgelist.txt            
t=10              
d=10 


for h in $H;
do 
	for v in $V;
	do  
	  time python3 womg --graph $g --virality $v --homophily $h --int_mode $i --docs $d --topics $t 
	done
done