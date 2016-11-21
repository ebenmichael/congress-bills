"""
Extract roll call data from votes folder as a (n_bills, n_senators) matrix and
save this along with associated meta data for the senators
"""

import json
import numpy as np
import os
import sys


def get_senator_data(votes_dir):
    """Make a map of senator id to column number in data matrix
    Input:
        votes_dir: string, path to a directory of votes for a congress
    Output:
        id_to_pos: dict, mapping from id to column in data matrix
        metadata: dict, mapping from id to name and party
    """
    def get_senators(vote_file):
        """Internal method to get senator ids and metadata from a file"""
        # open json file
        with open(vote_file) as f:
            vote_data = json.load(f)

        # get all the senators
        ids = set()
        metadata = {}
        for vote_type in vote_data["votes"].keys():
            for senator in vote_data["votes"][vote_type]:
                # skip the case that the VP breaks the tie
                if not type(senator) == dict:
                    continue
                if not senator["id"] in ids:
                    ids.add(senator["id"])
                    metadata[senator["id"]] = {k: v for k, v in senator.items()
                                               if k != "id"}
        return(ids, metadata)
    # get all of the senate files
    # actually HOUSE FILES
    years = [os.path.join(votes_dir, f) for f in os.listdir(votes_dir)]
    sub_dirs = [os.path.join(year, f) for year in years
                for f in os.listdir(year)
                if not os.path.isfile(f) and f[0] == "h"]
    # go through all of these files and get all of the senators
    ids = set()
    metadata = {}
    for bill in sub_dirs:
        bill_ids, bill_metadata = get_senators(os.path.join(bill, "data.json"))
        ids = ids.union(bill_ids)
        metadata.update(bill_metadata)
    # sort ids and make positions out of them
    id_to_pos = {id: i for i, id in enumerate(sorted(ids))}
    return(id_to_pos, metadata)


def get_vote(vote_file, id_to_pos):
    """Read a roll call vote json file and turn it into a vector
       Mapping from vote to number:
        Missing: -1
        Nay: 0
        Yea/Aye: 1
        Present: 2
        Not Voting: 3
    Input:
        vote_file: string, path to a json file with roll call votes from a bill
        id_to_pos: dict, mapping from id to column in data matrix
    Output:
        vote_vec: ndarray, length n_senators, votes as a vector
        bill_info: dict, mapping from {congress,number,category,chamber,type,
                   file_name} to their values
    """

    # open json file
    with open(vote_file) as f:
        vote_data = json.load(f)
    # only do cetrain kinds of votes
    black_list = ["leadership", "unknown"]
    if vote_data["category"] in black_list:
        return(None)
    # create the vector
    vote_vec = np.ones(len(id_to_pos)) * (-1)
    # mapping of vote type to number
    vote_map = {"Nay": 0, "No": 0, "Yea": 1, "Aye": 1,
                "Present": 2, "Not Voting": 3}

    for vote_type in vote_data["votes"]:
        for senator in vote_data["votes"][vote_type]:
            # skip the case where the VP breaks the tie
            if not type(senator) == dict:
                continue
            pos = id_to_pos[senator["id"]]
            vote_vec[pos] = vote_map[vote_type]

    bill_info = {"chamber": vote_data["chamber"],
                 "congress": vote_data["congress"],
                 "file_name": vote_file}
    if "amendment" in vote_data.keys():
        bill_info["type"] = vote_data["bill"]["type"]
        bill_info["category"] = "amendment"
        bill_info["number"] = vote_data["amendment"]["number"]
    elif "bill" in vote_data.keys():
        bill_info["type"] = vote_data["bill"]["type"]
        bill_info["category"] = "bill"
        bill_info["number"] = vote_data["bill"]["number"]
    elif "nomination" in vote_data.keys():
        bill_info["type"] = "nomination"
        bill_info["category"] = "nomination"
        bill_info["number"] = vote_data["nomination"]["number"]
    else:
        return(None)
    return(vote_vec, bill_info)


def get_senate_votes(votes_dir, id_to_pos):
    """Read senate roll call votes from a directory and turn into a matrix
    Input:
        votes_dir: string, path to a directory of votes for a congress
        id_to_pos: dict, mapping from id to column in data matrix
    Output:
        vote_mat: ndarray, shape (n_bills, n_senators), matrix of votes
        row_to_bill: dict, mapping from row in matrix to bill number, congress,
                     chamber, and category
    """
    # get all of the senate votes
    years = [os.path.join(votes_dir, f) for f in os.listdir(votes_dir)]
    sub_dirs = [os.path.join(year, f) for year in years
                for f in os.listdir(year)
                if not os.path.isfile(f) and f[0] == "h"]
    # make the matrix
    n_senators = len(id_to_pos)
    n_bills = len(sub_dirs)
    vote_mat = np.empty((n_bills, n_senators))

    row_to_bill = {}

    # sort by bill number
    sorted_sub_dirs = sorted(sub_dirs,
                             key=lambda x:
                             (int(os.path.split(os.path.split(x)[0])[1]),
                              int(os.path.split(x)[1][1:])))
    # iterate through the bills
    for i, bill in enumerate(sorted_sub_dirs):
        vote_file = os.path.join(bill, "data.json")
        # if we skipped the bill just get out of here
        output = get_vote(vote_file, id_to_pos)
        if output is None:
            continue
        vote_vec, bill_info = output
        vote_mat[i, :] = vote_vec
        row_to_bill[i] = bill_info

    return(vote_mat, row_to_bill)


def get_votes_and_data(votes_dir):
    """Get senator metadata, position mappings, and vote data as a matrix
    Input:
        votes_dir: string, path to a directory of votes for a congress
    Output:
        id_to_pos: dict, mapping from id to column in data matrix
        metadata: dict, mapping from id to name and party
        vote_mat: ndarray, shape (n_bills, n_senators), matrix of votes
        row_to_bill: dict, mapping from row in matrix to bill number, congress,
                     chamber, and category
    """
    # get position mappings and metadata
    id_to_pos, metadata = get_senator_data(votes_dir)
    # get the votes matrix
    vote_mat, row_to_bill = get_senate_votes(votes_dir, id_to_pos)

    return(id_to_pos, metadata, vote_mat, row_to_bill)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Point me to the votes data please!")
    elif len(sys.argv) == 2:
        votes_dir = sys.argv[1]
        output = get_votes_and_data(votes_dir)
        id_to_pos, metadata, vote_mat, row_to_bill = output
        # create a new directory for the data
        votes_dir = os.path.abspath(votes_dir)
        new_dir = os.path.join(os.path.split(votes_dir)[0], "combined_data")
        new_dir = os.path.relpath(new_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
            print("Creating new directory " + new_dir)
        # save everything
        with open(os.path.join(new_dir, "id_to_position.json"), "w") as f:
            json.dump(id_to_pos, f, sort_keys=True, indent=4)

        with open(os.path.join(new_dir, "senator_metadata.json"), "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=4)

        with open(os.path.join(new_dir, "row_to_bill.json"), "w") as f:
            json.dump(row_to_bill, f, sort_keys=True, indent=4)

        np.savetxt(os.path.join(new_dir, "votes.dat"), vote_mat)
    else:
        print("Too many arguments")
