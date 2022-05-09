#!/bin/bash
F_out="result_pubtator.txt"
F_list="index.txt"

echo -e "\n" >$F_out

i=1
while IFS= read -r line
do
 sleep 0.11s
 #line1=35313821
 #a=echo https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids=$line
 #curl \$a >>$F_out
 curl https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids=$line >>$F_out
 printf "$i \n"
 i=$[i+1]
done < "$F_list"

