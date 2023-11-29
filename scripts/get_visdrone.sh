#!/bin/bash
# VisDrone 2019 dataset for object detection in images https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?pli=1
# Unpack command: bash ./scripts/get_visdrone.sh

dsave='./visdrone/'
dzip='./' # unzip directory
dziproot='../../Datasets/UAV/VisDrone/'
froot='VisDrone2019-DET-'
ftrain='train'
fval='val'
ftest='test'

for f in $ftrain; do # $ftrain $fval $ftest; do
  echo 'Extracting' $froot$f '...'
  unzip -q $dziproot$froot$f'.zip' -d './temp/' # $dzip$froot$f'/'

  wait

  cp -r './temp/'$froot$f'/images/' $dsave'images/'$f'/'
  cp -r './temp/'$froot$f'/annotations/' $dsave'labels_raw/'$f'/'
done


wait # finish background tasks


# rm -r './temp/' & 
echo 'Done.'
###
