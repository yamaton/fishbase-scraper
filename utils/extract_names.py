#!/usr/bin/env python
"""
The program extracts a list of species from the FishBase page containing names.
The HTML must have the structure: https://fishbase.se/ListByLetter/ScientificNamesZ.htm.
"""

import bs4
import argparse

def extract(content: str) -> list[str]:
    """Extract a list of names from string in HTML
    """
    soup = bs4.BeautifulSoup(content, features="html.parser")
    xs = soup.table.find_all("a")
    res = [x.string for x in xs]
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input file")
    args = parser.parse_args()
    with open(args.file) as f:
        content = f.read()

    xs = extract(content)
    for x in xs:
        print(x)
