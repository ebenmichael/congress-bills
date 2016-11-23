# congress-bills
Predicting roll call votes for the senate. Project for 241a

## Getting the data
The data is from [GovTrack.us](https://www.govtrack.us/developers/data) To download the data with rsync use the command
```
rsync -avz --delete --delete-excluded --exclude **/text-versions/ 	govtrack.us::govtrackdata/congress/110/votes .
```
This gets the roll call votes for the 110th congress.
To get metadata and text for all of the bills in the 110th congress (not just the ones voted on) run
```
rsync -avz --delete --delete-excluded --exclude **/text-versions/ 	govtrack.us::govtrackdata/congress/110/bills .
```
and to get the text for all of the house ammendments in the 110th congress run
```
rsync -avz --delete --delete-excluded --exclude **/text-versions/ 	govtrack.us::govtrackdata/congress/110/amendments/hamdt .
```
## Processing the data
To get the data into a more usable form run
```
./data_script.sh
```
This runs
```
python read_data.py votes
```

which creates a directory `combined_data/` with the following files
- `votes.dat`: voting data as a (n_bills, n_senators) matrix where votes are coded as 
    - Missing (e.g. the senator died or wasn't elected yet): -1
    - Nay: 0
    - Yea: 1
    - Present: 2
    - Not Voting: 3

- `id_to_position.json`: a mapping from senator id to column in `votes.dat`
- `row_to_bill.json`: a mapping from row in `votes.dat` to bill congress, chamber, category, and number
- `senator_metadata.json`: a mapping from senator id to name and party

It also runs
```
python text_to_features.py bills hamdt combined_data/row_to_bill.json
```
which extracts the text from the bills and amendments into a corpus and get a sparse bag of words feature matrix for the corpus run
Finally it runs
```
python load_caucus_data.py  caucus/MC_110.csv caucus/membership_110.csv combined_data/id_to_position.json combined_data/senator_metadata.json
```