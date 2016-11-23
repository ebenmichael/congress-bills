#!/bin/bash


echo "Reading vote data..."
python read_data.py votes 

echo "Creating a bag of words matrix for the bill/ammendment text..."
python text_to_features.py bills hamdt combined_data/row_to_bill.json

echo "Creating a representative by caucus matrix..."
python load_caucus_data.py  caucus/MC_110.csv caucus/membership_110.csv combined_data/id_to_position.json combined_data/senator_metadata.json
