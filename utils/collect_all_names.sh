#!/usr/bin/env bash

## Scrape all fish names FishBase pages from A to Z
##
## https://fishbase.se/ListByLetter/ScientificNamesA.htm
## ...
## https://fishbase.se/ListByLetter/ScientificNamesZ.htm
##
## And returns a list of all species names available in FishBase
##
##
## Usage:
##     $ bash utils/collect_all_names.sh > ScientificNamesAll.txt
##
## Requirements:
##     - GNU parallel
##         (Install via `sudo apt install parallel` in Ubuntu.)
##

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rm -f *.htm
echo {A..Z} | tr -s ' ' '\n' | parallel wget 'https://fishbase.se/ListByLetter/ScientificNames{}.htm'
ls *.htm | parallel python "$BASEDIR/extract_names.py" | cat | sort
rm -f *.htm
