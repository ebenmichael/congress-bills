# congress-bills
Predicting roll call votes for the senate. Project for 241a

## Getting the data
The data is from [GovTrack.us](https://www.govtrack.us/developers/data) To download the data with rsync use the command
```
rsync -avz --delete --delete-excluded --exclude **/text-versions/ 	govtrack.us::govtrackdata/congress/113/bills .
```
This gets the roll call votes for the 113th congress.