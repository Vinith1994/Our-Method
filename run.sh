#!/bin/bash
#rm -rf gen_graphs
#unzip give.zip
echo $1
for filepath in original_datasets/$1_dataset/*.edgelist
do
  echo $filepath
  filename=$(basename -s .edgelist "$filepath")
  #echo $filename
  dname=$(dirname "$filepath")
  #echo $dname
  fullpath="$dname/$filename.rolelist"
  #echo $fullpath
  #cd struc2vec && python src/main.py --input ../$filepath --output tmp.emb && cd ..
  #cd struc2vec && python src/main.py --input ../$filepath --output tmp.emb --workers 64 --OPT1 True --OPT2 True && cd ..
  python3 test.py $filename $fullpath $1 "without_extra" "with_opt"
done
#zip -r house_pickles.zip send_pickles/
#echo "house pickles zip" | mutt -a "house_pickles.zip" -s "house pickles zip" -c suresh43@purdue.edu -y
