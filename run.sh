#!/bin/bash
#rm -rf gen_graphs
#unzip give.zip
for filepath in cavemen_dataset/*.edgelist
do
  echo $filepath
  filename=$(basename -s .edgelist "$filepath")
  #echo $filename
  dname=$(dirname "$filepath")
  #echo $dname
  fullpath="$dname/$filename.rolelist"
  #echo $fullpath
  cd struc2vec && python src/main.py --input ../$filepath --output tmp.emb && cd ..
  python3 test.py $filename $fullpath
done
zip -r send.zip send_pickles/
echo "Send pickle zip" | mutt -a "send.zip" -s "Send pickle zip" -c suresh43@purdue.edu -y
