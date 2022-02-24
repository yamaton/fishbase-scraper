#!/usr/bin/env python

"""
Check if names in the given list are found in FishBase. Suggest correction, otherwise.

Recommended dependencies:
    - pathos
    Install via either `pip install pathos` or `conda install -c conda-forge pathos`.

Usage:
    $ python check_names.py merged_list.txt

    To add up to 4 mutations for possible matches (which reduces "?????" outputs),
    $ python check_names.py --distance 4 merged_list.txt


[NOTE] The algorithm internally uses two versions: a generative algorithm is
used for distance <= 2 and Levenshtein distance is used otherwise.
Two algorithms are *inconsistent* in their definitions of "distance";
a single mutation (or edit) includes transposition in the former, but not in the latter.
Replace Levenshtein with Damerau–Levenshtein to fix this inconsistencies in distances.
"""

import argparse
import gzip
import logging
import pathlib
import subprocess
import itertools as it
from typing import Union, Set

BASEDIR = pathlib.Path(__file__).resolve().parent
REFERENCE = BASEDIR / "ScientificNamesAll.txt"
REFERENCE_NCBI =  BASEDIR / "ncbi_taxdump" / "ncbi_names.txt.gz"

# unknown output
UNKNOWN = "?????"

# set logging level
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO
)


def levenshtein(s1: str, s2: str) -> int:
    """Calculate Levenshetein distance of two strings
    Copied from https://rosettacode.org/wiki/Levenshtein_distance#Iterative_2
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]


def mutate(word: str) -> Set[str]:
    """Mutate string"""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = (l + r[1:] for l, r in splits if r)
    transposes = (l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1)
    inserts = (l + c + r for l, r in splits for c in letters)
    replaces = (l + c + r[1:] for l, r in splits if r for c in letters)
    iterable = it.chain(deletes, transposes, inserts, replaces)
    res = {x for x in iterable if len(x.split()) > 1}
    return res


def update(bag: Set[str]) -> Set[str]:
    return {w_new for w in bag for w_new in mutate(w)}


def download():
    p = BASEDIR / "utils" / "collect_all_names.sh"
    subprocess.run(f"bash {p.as_posix()} > {REFERENCE.as_posix()}", shell=True)


def get_suggestion(sample: str, allnames: Set[str], num_attempts: int) -> str:
    if num_attempts < 3:
        if sample in allnames:
            return sample

        bag = {sample}
        for _ in range(num_attempts):
            bag = update(bag)
            for x in bag:
                if x in allnames:
                    return x
        return UNKNOWN

    candidate = min(allnames, key=lambda x: levenshtein(sample, x))
    return candidate if levenshtein(sample, candidate) <= num_attempts else UNKNOWN


def normalize(s: str) -> str:
    if s.startswith("[") or s.startswith("?"):
        return s
    genus, species = s.split()
    genus = genus.capitalize()
    species = species.lower()
    return f"{genus} {species}"


def load_names(p: Union[str, pathlib.Path]) -> Set[str]:
    p = pathlib.Path(p)
    f = gzip.open(p, "rt") if p.suffix == ".gz" else p.open()
    names = {' '.join(line.strip().lower().split()) for line in f.readlines() if line.strip()}
    f.close()
    return names


if __name__ == "__main__":
    if not REFERENCE.exists():
        print(f"File not found: {REFERENCE}")
        print(f"Running `bash utils/collect_all_names > ScientificNamesAll.txt` to create the list of all fish names.")
        download()
        print()
        print("Saved all scientific names from FishBase as ScientificNamesAll.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input file")
    parser.add_argument(
        "--distance",
        type=int,
        default=2,
        help=f"""Say '{UNKNOWN}' if a match is not found within this distance.""",
    )
    parser.add_argument(
        "--ncbi",
        help="Check against NCBI taxonomy dump if failed to match Fishbase.",
        action='store_true',
    )
    args = parser.parse_args()
    num_mutations = args.distance
    use_ncbi_taxdump = args.ncbi

    allnames = load_names(REFERENCE)
    names = load_names(args.file)
    ncbi_names = set()
    if use_ncbi_taxdump:
        if REFERENCE_NCBI.exists():
            ncbi_names = load_names(REFERENCE_NCBI)
        else:
           logging.warning(f"NCBI taxdump is missing at {REFERENCE_NCBI}")
           logging.warning("Consider getting from https://github.com/yamaton/fishbase-scraper/raw/main/ncbi_taxdump/ncbi_names.txt.gz")

    not_found = sorted(names - allnames)
    ## Report summary
    num_entries = len(names)
    matched = len(names & allnames)
    unmatched = num_entries - matched
    logging.info(f"Found exact matching in FishBase:  {matched} out of {num_entries}")
    logging.info(f"Suggested names for the rest of {len(not_found)} entries:")

    allgenus, allspecies = zip(*[x.split() for x in allnames])
    allgenus = set(allgenus)
    allspecies = set(allspecies)

    ## Suggest based on genus-species pair similarity
    try:
        from pathos.multiprocessing import ProcessingPool as Pool
        pool = Pool()
        suggestions = pool.map(lambda name: get_suggestion(name, allnames, num_mutations), not_found)
    except ImportError:
        logging.warning("Package pathos is not found. Running without parallelization...")
        suggestions = (get_suggestion(name, allnames, num_mutations) for name in not_found)
    for name, suggestion in zip(not_found, suggestions):
        if suggestion == UNKNOWN:
            if name in ncbi_names:
                suggestion = "[FOUND in NCBI taxdump; unavailable in FishBase]"
            else:
                genus, species = name.split()
                if genus in allgenus and species in allspecies:
                    suggestion = "[OK by words; not found in FishBase]"
        print(f"{normalize(name)}\t-->\t{normalize(suggestion)}")
