"""
Merge files and keep unique names

Assume each file has genus-speies names in a single column.

Usage:
    python merge_and_keep_unique_names.py file1 file2 file3 > merged_list.txt

    The number of arguments can be arbitrary many.

"""

import argparse
import logging
from typing import Set


# logging format and level
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO
)


def get_genus_species(filename: str) -> Set[str]:
    """Extract space-separated word pairs that are meant by (Genus, Species) names"""
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

    result = set()
    for linenum, line in enumerate(lines):
        words = line.split()
        if len(words) == 1:
            logging.warning(f"Is this a scientific name?  {line}")
            result.add(line)
            continue

        if len(words) > 2:
            name = " ".join(words)
            logging.warning(f"Irregular scientific name?  {name}")
            result.add(name)
            continue

        genus, species = words
        if not genus.istitle():
            logging.warning(
                f"[{filename}: line {linenum + 1}] Correct genus name: {genus} --> {genus.title()}"
            )
            genus = genus.title()
        if not species.islower():
            logging.warning(
                f"[{filename}: line {linenum + 1}] Correct species name {species} --> {species.lower()}"
            )
            species = species.lower()
        genus_species = f"{genus} {species}"
        result.add(genus_species)

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", metavar="<file>", help="""Input files in a row.""")
    args = parser.parse_args()
    xs = [get_genus_species(p) for p in args.file]
    s = sorted(set.union(*xs))
    for genus_species in s:
        print(genus_species)
