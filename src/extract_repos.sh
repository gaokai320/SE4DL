# extract DL repositories using tensorflow, keras, and pytorch`
# login WoC da4 server /data/play
mkdir /data/play/dl_repos
cd /data/play/dl_repos
for i in {0..127}; do zcat /da5_data/basemaps/gz/c2PtabllfPkgFullT$i.s | grep -E "\;tensorflow|\;torch|\;keras" >> dl_project; echo $i done; done
cat dl_project | cut -d\; -f2 | sort -u > project_unique
cat project_unique | while read line; do  t=`echo $line | ~/lookup/getValues -f p2c | cut -d\; -f2 | ~/lookup/showCnt commit 1 | sort -t\; -k 2 | tail -1 | cut -d\; -f1`; echo $line\;$t; done > p2lc 2>p2lc_err
cat p2lc | while read line; do p=`echo $line | cut -d\; -f1`; c=`echo $line | cut -d\; -f2`; t=`echo $c | ~/lookup/showCnt commit | cut -d\; -f2 | ~/lookup/showCnt tree | grep ";README.md$"`; if [ $t ]; then echo "$p;$t"; fi; done > p2r 2>p2r_err
cat p2r | while read line; do p=`echo $line | cut -d\; -f1`; sha=`echo $line | cut -d\; -f3`; content=`echo $sha | ~/lookup/showCnt blob`; if [ "$content" ]; then echo "$content" > readmes/$p; fi; done 2>blob_err
tar -zcvf readmes.tar.gz readmes/