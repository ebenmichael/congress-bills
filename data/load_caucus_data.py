#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
import sys

def caucus_id_to_row(members, id_to_row, senator_metadata):
    """Get a mapping from caucus data id numbers to rows in our matrix"""
    c_name_to_id = {}
    for idx, row in members.iterrows():
        c_name_to_id[(row["name"], row["state"])] = row["idno"]
    # map govtrack.us ids to caucus data ids
    c_id_to_row = {}
    missing = {}
    for hid in senator_metadata.keys():
        disp_name = senator_metadata[hid]["display_name"]
        disp_name = disp_name.replace("á", "a")
        state = state_to_num[senator_metadata[hid]["state"]]
        # get the name before the parens
        paren_pos = disp_name.find("(")
        if paren_pos != -1:
            disp_name = disp_name[: paren_pos - 1]
        # find the comma if it's lastname, firstname
        last_pos = disp_name.find(",")
        if last_pos != - 1:
            disp_name = disp_name[: last_pos]
        # get the caucus id, keep getting rid of letters until it works
        succeeded = False
        while not succeeded:
            try:
                if len(disp_name) == 0:
                    succeeded = True
                c_id = c_name_to_id[(disp_name.upper(), state)]
                succeeded = True
            except KeyError:
                disp_name = disp_name[:-1]
        if len(disp_name) == 0:
            missing[hid] = senator_metadata[hid]
        # get the row
        row = id_to_row[hid]
        c_id_to_row[c_id] = row
    c_id_to_row["14256"] = id_to_row['M000725']
    c_id_to_row["20350"] = id_to_row['D000599']
    c_id_to_row["20141"] = id_to_row["D000597"]
    c_id_to_row["20346"] = id_to_row["M001151"]
    c_id_to_row["20529"] = id_to_row["G000553"]
    c_id_to_row["29143"] = id_to_row["J000174"]
    c_id_to_row["29336"] = id_to_row["D000600"]
    c_id_to_row["29709"] = id_to_row["S000030"]
    c_id_to_row['C001073'] = id_to_row['C001073']
    c_id_to_row["N000159"] = id_to_row["N000159"]
    return(c_id_to_row)


def permute_rows(membership, c_id_to_row):
    """Permute the caucus membership matrix to match our indices"""
    new_m = np.empty((membership.shape[0], membership.shape[1]))
    # iterate over the data
    for c_id, caucuses in membership.iterrows():
        row = c_id_to_row[str(c_id)]
        new_m[row, :] = caucuses.values
    return(new_m)

# mapping from state abbrev to number
state_to_num = {'WA': 73, 'DE': 11, 'DC': 55, 'WI': 25, 'WV': 56, 'HI': 82,
                'FL': 43, 'WY': 68, 'NH': 4, 'NJ': 12, 'NM': 66, 'TX': 49,
                'LA': 45, 'NC': 47, 'ND': 36, 'NE': 35, 'TN': 54, 'NY': 13,
                'PA': 14, 'CA': 71, 'NV': 65, 'VA': 40, 'CO': 62, 'AK': 81,
                'AR': 42, 'VT': 6, 'IL': 21, 'GA': 44, 'IN': 22, 'IA': 31,
                'OK': 53, 'AZ': 61, 'ID': 63, 'CT': 1, 'ME': 2, 'MD': 52,
                'MA': 3, 'OH': 24, 'UT': 67, 'MO': 34, 'MN': 33, 'MI': 23,
                'RI': 5, 'KS': 32, 'MT': 64, 'MS': 46, 'SC': 48, 'KY': 51,
                'OR': 72, 'SD': 37, 'AL': 41}

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python load_caucus_data.py caucus/MC_110.csv " +
              "caucus/membership_110.csv combined_data/id_to_pos.json " +
              "combined_data/senator_metadata.json")
    else:
        members = pd.read_csv(sys.argv[1])
        membership = pd.read_csv(sys.argv[2], index_col=0)
        with open(sys.argv[3]) as f:
            id_to_row = json.load(f)
        with open(sys.argv[4]) as f:
            senator_metadata = json.load(f)
        c_id_to_row = caucus_id_to_row(members, id_to_row, senator_metadata)
        Y = permute_rows(membership, c_id_to_row)
        # save Y
        np.savetxt("combined_data/membership.dat", Y)
