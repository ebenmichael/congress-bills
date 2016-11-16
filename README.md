# congress-bills
Predicting roll call votes for the senate. Project for 241a

## Getting the data
The data is from [GovTrack.us](https://www.govtrack.us/developers/data) To download the data with rsync use the command
```
rsync -avz --delete --delete-excluded --exclude **/text-versions/ 	govtrack.us::govtrackdata/congress/110/votes .
```
This gets the roll call votes for the 110th congress. To get the data into a more usable form run

```
python read_data.py votes
```

This will create a directory `combined_data/` with the following files
- `votes.dat`: voting data as a (n_bills, n_senators) matrix where votes are coded as 
    - Missing (e.g. the senator died or wasn't elected yet): -1
    - Nay: 0
    - Yea: 1
    - Present: 2
    - Not Voting: 3

- `id_to_position.json`: a mapping from senator id to column in `votes.dat`
- `row_to_bill.json`: a mapping from row in `votes.dat` to bill congress, chamber, category, and number
- `senator_metadata.json`: a mapping from senator id to name and party

To get metadata and text for all of the bills in the 110th congress (not just the ones voted on) run
```
rsync -avz --delete --delete-excluded --exclude **/text-versions/ 	govtrack.us::govtrackdata/congress/110/bills .
```