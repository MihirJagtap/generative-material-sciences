#!/bin/bash

tar zxvf cdvae-main-32BS.tar.gz 

cd cdvae-main

source .env

export WANDB_API_KEY=7982a8264586c151109b3022859414796064841a

python cdvae/run.py data=mp_20 expname=mp_20 model.predict_property=True 

cd .. 

unique_id=$(date +"%Y%m%d%H%M%S")


tar -czvf cdvae-train-32BS-output-$unique_id.tar.gz cdvae-main 

rm -r cdvae-main