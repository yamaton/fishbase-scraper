#!/usr/bin/env python

"""
Check if names in the given list are found in FishBase. Suggest correction, otherwise.

Usage:
$ python check_names.py merged_list.txt

To change degree of suggestion vs "?????",


"""

import argparse
import pathlib
import subprocess

BASEDIR = pathlib.Path(__file__).resolve().parent
REFERENCE = BASEDIR / "ScientificNamesAll.txt"

def levenshtein(s1: str, s2: str) -> int:
    """Calculate Levenshetein distance of two strings

    Copied from https://rosettacode.org/wiki/Levenshtein_distance#Iterative_2
    """
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]


def download():
    p = BASEDIR / "utils" / "collect_all_names.sh"
    subprocess.run(f"bash {p.as_posix()} > {REFERENCE.as_posix()}", shell=True)


def get_suggestion(sample: str, allnames: set[str]) -> str:
    return min(allnames, key=lambda x: levenshtein(sample, x))


if __name__ == "__main__":
    if not REFERENCE.exists():
        print(f"File not found: {REFERENCE}")
        print(f"Running `bash utils/collect_all_names > ScientificNamesAll.txt` to create the list of all fish names.")
        download()
        print()
        print("Saved all scientific names from FishBase as ScientificNamesAll.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input file")
    parser.add_argument("--distance", type=int, default=5, help="""Say 'Not Found' if the edit distance is equal or greater than this threshold.""")
    args = parser.parse_args()

    with REFERENCE.open() as f:
        allnames = {line.strip() for line in f.readlines() if line.strip()}

    with open(args.file) as f:
        names = {line.strip() for line in f.readlines() if line.strip()}

    not_found = sorted(names - allnames)
    ## Report summary
    num_entries = len(names)
    matched = len(names & allnames)
    unmatched = num_entries - matched
    print(f"Matched:  {matched}/{num_entries}")
    print()

    allgenus, allspecies = zip(*[x.split() for x in allnames])
    allgenus = set(allgenus)
    allspecies = set(allspecies)

    ## Suggest based on genus-species pair similarity
    for name in not_found:
        suggestion = min(allnames, key=lambda x: levenshtein(name, x))
        if levenshtein(name, suggestion) >= args.distance:
            suggestion = "?????"
            genus, species = name.split()
            if genus in allgenus and species in allspecies:
                suggestion = "[good by words; not found in FishBase]"
        print(f"{name}\t-->\t{suggestion}")

