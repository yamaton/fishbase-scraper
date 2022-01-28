"""
Merge files and keep unique names

Assume each file has genus-speies names in a single column.

Usage:
python merge_and_keep_unique_names.py file1 file2 > merged_list.txt

"""

import argparse
import logging


def get_genus_species(filename: str) -> set[str]:
    """Extract space-separated word pairs that are meant by (Genus, Species) names"""
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

    result = set()
    for linenum, line in enumerate(lines):
        words = line.split()
        assert len(words) == 2
        genus, species = words
        if not genus.istitle():
            logging.error(
                f"[{filename}: line {linenum + 1}] Correct genus name: {genus} --> {genus.title()}."
            )
            genus = genus.title()
        if not species.islower():
            logging.error(
                f"[{filename}: line {linenum + 1}] Correct species name {species} --> {species.lower()}."
            )
            species = species.lower()
        genus_species = f"{genus} {species}"
        result.add(genus_species)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="""Input file1""")
    parser.add_argument("file2", help="""Input file1""")
    args = parser.parse_args()
    s1 = get_genus_species(args.file1)
    s2 = get_genus_species(args.file2)
    s = sorted(s1 | s2)
    for genus_species in s:
        print(genus_species)
