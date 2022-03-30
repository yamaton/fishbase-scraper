#!/usr/bin/env python

"""
Check if names in the given list are found in FishBase. Suggest correction, otherwise.

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
import collections
import csv
import gzip
import json
import logging
import pathlib
import subprocess
import sys
from urllib.error import HTTPError
import urllib.parse
import urllib.request
import itertools as it
from typing import Any, Dict, Iterable, Optional, Union, Set, List

BASEDIR = pathlib.Path(__file__).resolve().parent
REFERENCE = BASEDIR / "ScientificNamesAll.txt"
REFERENCE_NCBI =  BASEDIR / "ncbi_taxdump" / "ncbi_names.txt.gz"

# unknown output representation
UNKNOWN = "?????"
PROVIDER_FISHBASE = "FishBase"
PROVIDER_WORMS = "WoRMS"
PROVIDER_NCBI_TAXDUMP = "NCBI Taxdump"
PROVIDER_SPELL_CORRECTOR = "misspelling"
NO_PROVIDER = ""

# logging format and level
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO
    # level=logging.DEBUG
)

# Data structure for name-correction result
#    `input`:    scientific name entered by user
#    `output`:   scientific name corrected. Can be identical to `input`
#    `provider`: correction provider. Can be "FishBase", "WoRMS", ...
Result = collections.namedtuple('Result', ['input', 'output', 'provider'])


def get_suggestion_worm(scientific_name: str) -> str:
    """
    Use WoRMS webservice: /AphiaRecordsByName/{ScientificName}
    to correct a scientific name.
    https://www.marinespecies.org/rest/

    >>> worms_make_request("foo bar")
    "?????"

    >>> worms_make_request("Paraplotosus albilabrus")
    "Paraplotosus albilabris"

    >>> worms_make_request("Parupeneus cinnabarinus")
    "Parupeneus heptacanthus"

    """

    def lint(scientific_name: str) -> str:
        return " ".join(scientific_name.strip().split())


    # curl -X GET "https://www.marinespecies.org/rest/AphiaRecordsByName/Parupeneus%20cinnabarinus?like=true&marine_only=true&offset=1" -H  "accept: */*"
    query = urllib.parse.quote(lint(scientific_name))
    url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{query}"
    values = {
        "like": True,
        "marine_only": True,
        "offset": 1,
    }
    data = urllib.parse.urlencode(values)
    full_url = url + "?" + data
    logging.debug(f"{full_url = }")

    try:
        with urllib.request.urlopen(full_url) as response:
            raw = response.read()
        logging.debug(f"{raw = }")
    except HTTPError:
        logging.error(f"HTTPError from WoRMS: {scientific_name}")
        return UNKNOWN

    if not raw:
        return UNKNOWN

    d = json.loads(raw)[0]
    if "status" in d:
        status = d["status"]
        if status == "accepted":
            return d["scientificname"]
        elif status in ["unaccepted", "alternate representation"]:
            return d["valid_name"]
        else:
            logging.warning(f"Got an unknown status ({status}): {scientific_name}")
            return UNKNOWN

    logging.warning(f"WoRMS's response lacks \"status\": {d}")
    return UNKNOWN



def to_chunks(xs: Iterable[Any], chunksize: int) -> List[List[Any]]:
    """
    Split an iterable into fixed-sized chunks

    >>> to_chunks(range(10), 3)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    res = []
    chunk = []
    for x in xs:
        chunk.append(x)
        if len(chunk) == chunksize:
            res.append(chunk)
            chunk = []

    if chunk:
        res.append(chunk)
    return res


def get_suggestion_worm_batch(scientific_names: List[str], chunksize=100) -> Dict[str, str]:
    """
    Use WoRMS webservice: /AphiaRecordsByNames

    """
    res = dict()
    chunks = to_chunks(scientific_names, chunksize)
    for chunk in chunks:
        d = _get_suggestion_worm_batch(chunk)
        res.update(d)
    return res


def _get_suggestion_worm_batch(scientific_names: List[str]) -> Dict[str, str]:
    """
    Use WoRMS webservice: /AphiaRecordsByNames
    to correct a scientific name.
    https://www.marinespecies.org/rest/

    >>> _get_suggestion_worm_batch(["foo bar", "Paraplotosus albilabrus", "Parupeneus cinnabarinus"])
    ["?????", "Paraplotosus albilabris", "Parupeneus heptacanthus"]

    """
    assert len(scientific_names) <= 500
    zeroinfo = {s: UNKNOWN for s in scientific_names}

    def lint(scientific_name: str) -> str:
        return " ".join(scientific_name.strip().split())

    queries = ["scientificnames[]=" + urllib.parse.quote(lint(s)) for s in scientific_names]
    query = "&".join(queries)

    # curl -X GET "https://www.marinespecies.org/rest/AphiaRecordsByNames?scientificnames[]=foo%20bar&scientificnames[]=Paraplotosus%20albilabrus&scientificnames[]=Parupeneus%20cinnabarinus&like=false&marine_only=true" -H  "accept: */*"
    url = f"https://www.marinespecies.org/rest/AphiaRecordsByNames?{query}"
    values = {
        "like": True,
        "marine_only": True,
    }
    data = urllib.parse.urlencode(values)
    full_url = url + "&" + data
    logging.debug(f"{full_url = }")

    try:
        with urllib.request.urlopen(full_url) as response:
            raw = response.read()
        logging.debug(f"{raw = }")
    except HTTPError:
        logging.error(f"HTTPError from WoRMS ({len(scientific_names)} entries)")
        return zeroinfo

    if not raw:
        logging.error(f"WoRMS returned empty value.")
        return zeroinfo

    res = dict()
    xs = json.loads(raw)
    if len(xs) != len(scientific_names):
        logging.error("Number of responses do not agree with the number of queries.")
        return zeroinfo

    for name, items in zip(scientific_names, xs):
        if not items:
            res[name] = UNKNOWN
            continue

        d = items[0]
        if "status" in d:
            status = d["status"]
            if status == "accepted":
                res[name] = d["scientificname"]
            elif status in ["unaccepted", "alternate representation"]:
                res[name] = d["valid_name"]
            else:
                logging.warning(f"Got an unknown status ({status}): {name}")
                res[name] = UNKNOWN
        else:
            logging.warning(f"WoRMS's response lacks \"status\": {d}")
            res[name] = UNKNOWN
    return res


def _levenshtein(s1: str, s2: str) -> int:
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


def _mutate(word: str) -> Set[str]:
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


def _update(bag: Set[str]) -> Set[str]:
    return {w_new for w in bag for w_new in _mutate(w)}


def download_all_fishbase_names():
    p = BASEDIR / "utils" / "collect_all_names.sh"
    subprocess.run(f"bash {p.as_posix()} > {REFERENCE.as_posix()}", shell=True)


def correct_spelling(sample: str, corpus: Set[str], num_attempts: int) -> str:
    if num_attempts < 3:
        if sample in corpus:
            return sample

        bag = {sample}
        for _ in range(num_attempts):
            bag = _update(bag)
            for x in bag:
                if x in corpus:
                    return x
        return UNKNOWN

    candidate = min(corpus, key=lambda x: _levenshtein(sample, x))
    return candidate if _levenshtein(sample, candidate) <= num_attempts else UNKNOWN


def normalize(s: str) -> str:
    """Normalize scientific name `s`. Don't change otherwise.
    """
    def is_scientific_name(s: str) -> bool:
        if s.startswith("[") or s.startswith("?"):
            return False

        if len(s.split()) != 2:
            logging.warning(f"Irregular name?: {s}")
            return False

        return True

    if not is_scientific_name(s):
        return s

    genus, species = s.split()
    genus = genus.capitalize()
    species = species.lower()
    return f"{genus} {species}"


def load_names(p: Union[str, pathlib.Path]) -> List[str]:
    p = pathlib.Path(p)
    f = gzip.open(p, "rt") if p.suffix == ".gz" else p.open()
    names = [' '.join(line.strip().lower().split()) for line in f.readlines() if line.strip()]
    f.close()
    return names


class Corrector(object):
    def __init__(self, corpus: Set[str], ncbi_corpus: Set[str], num_spelling_mutation: int, num_cpus: Optional[int]=None) -> None:
        self.corpus = corpus
        self.ncbi_corpus = ncbi_corpus
        self.num_mutation = num_spelling_mutation
        self.num_cpus = num_cpus

    def run(self, s: str) -> Result:
        """
        Check scientific name
        """
        if s in self.corpus:
            return Result(s, s, PROVIDER_FISHBASE)

        result_spell = correct_spelling(s, self.corpus, self.num_mutation)
        if result_spell != UNKNOWN:
            return Result(s, result_spell, PROVIDER_SPELL_CORRECTOR)

        result_worms = get_suggestion_worm(s)
        if result_worms != UNKNOWN:
            return Result(s, result_worms, PROVIDER_WORMS)

        if s in self.ncbi_corpus:
            return Result(s, s, PROVIDER_NCBI_TAXDUMP)

        return Result(s, UNKNOWN, NO_PROVIDER)


    def run_many(self, ss: Iterable[str]) -> Dict[str, Result]:
        """
        Check scientific names as batch
        """
        ss = list(ss)
        ans = dict()
        total = len(ss)
        for s in ss:
            if s in self.corpus:
                ans[s] = Result(s, s, PROVIDER_FISHBASE)
        sys.stderr.write(f"Progress: {len(ans):>4}/{total}:                                                            \r")

        rest = [s for s in ss if s not in ans]
        try:
            import pathos.multiprocessing
            with pathos.multiprocessing.ProcessPool(nodes=self.num_cpus) as p:
                results_spell = p.map(lambda s: correct_spelling(s, self.corpus, self.num_mutation), rest)
        except ImportError:
            logging.info("pathos is missing; Install pathos to enable multiprocessing...")
            results_spell = [correct_spelling(s, self.corpus, self.num_mutation) for s in rest]

        for s, result_spell in zip(rest, results_spell):
            if result_spell != UNKNOWN:
                ans[s] = Result(s, result_spell, PROVIDER_SPELL_CORRECTOR)

        rest = [s for s in ss if s not in ans]
        results_worms_batch = get_suggestion_worm_batch(rest)
        for s, result_worms in results_worms_batch.items():
            if result_worms != UNKNOWN:
                ans[s] = Result(s, result_worms, PROVIDER_WORMS)
            elif s in self.ncbi_corpus:
                ans[s] = Result(s, s, PROVIDER_NCBI_TAXDUMP)
            else:
                ans[s] = Result(s, UNKNOWN, NO_PROVIDER)
        return ans


def report(results: Iterable[Result]):
    """
    Report results to standard output
    """
    xs = list(results)
    num_entries = len(xs)
    matched = sum(1 for x in xs if x.provider == PROVIDER_FISHBASE)
    logging.info(f"Found exact matching in FishBase:  {matched} out of {num_entries}")

    not_in_fishbase = [x for x in xs if x.provider != PROVIDER_FISHBASE]
    logging.info(f"Suggested names for the rest of {len(not_in_fishbase)} entries:")

    for x in not_in_fishbase:
        print(f"{x.provider:12}: {normalize(x.input):30}\t-->\t{normalize(x.output)}")


def to_csv(results: Iterable[Result], filename: Union[str, pathlib.Path]):
    # fieldnames = ("original", "corrected", "provider")
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(results)


if __name__ == "__main__":
    if not REFERENCE.exists():
        logging.info(f"File not found: {REFERENCE}")
        logging.info(f"Running `bash utils/collect_all_names > ScientificNamesAll.txt` to create the list of all fish names.")
        download_all_fishbase_names()
        logging.info("Saved all scientific names from FishBase as ScientificNamesAll.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input file")
    parser.add_argument(
        "-o", "--output",
        metavar="<file>",
        help="Output file",
        default=None,
    )
    parser.add_argument(
        "--distance",
        type=int,
        metavar="<int>",
        default=2,
        help=f"""Say '{UNKNOWN}' if a match is not found within this distance. (default: %(default)s)""",
    )
    parser.add_argument(
        "--ncbi",
        help="Check against NCBI taxonomy dump if failed to find in Fishbase.",
        action='store_true',
    )
    parser.add_argument(
        "--num_cpus",
        help="Number of CPU cores to use; set -1 to use all. (default: %(default)s)",
        type=int,
        metavar="<int>",
        default=-1,
    )
    parser.add_argument(
        "--fast",
        help="Run faster by processing as a batch",
        action="store_true",
    )
    args = parser.parse_args()
    output_filename = args.output
    num_cpus = args.num_cpus if args.num_cpus > 0 else None
    num_mutations = args.distance
    use_ncbi_taxdump = args.ncbi

    logging.debug("Loading all scientific names in FishBase")
    corpus = set(load_names(REFERENCE))
    names = load_names(args.file)

    logging.debug("Loadiing NCBI taxdump...")
    ncbi_corpus = set()
    if use_ncbi_taxdump:
        logging.info("NCBI taxdump is currently hard-disabled due to issue with multiprocessing.")
        # if REFERENCE_NCBI.exists():
        #     ncbi_corpus = set(load_names(REFERENCE_NCBI))
        # else:
        #    logging.warning(f"NCBI taxdump is missing at {REFERENCE_NCBI}")
        #    logging.warning("Consider getting from https://github.com/yamaton/fishbase-scraper/raw/main/ncbi_taxdump/ncbi_names.txt.gz")
    logging.debug("... Done loading NCBI taxdump")


    machine = Corrector(corpus, ncbi_corpus, num_mutations, num_cpus)
    total = len(names)

    if args.fast:
        res_dict = machine.run_many(names)
        results = [res_dict[name] for name in names]
    else:
        results: List[Result] = []
        for i, name in enumerate(names, 1):
            sys.stderr.write(f"Progress: {i:>4}/{total}: {name}                                                        \r")
            res = machine.run(name)
            results.append(res)

    logging.debug(f"{[x.input for x in results]}")
    report(results)
    if output_filename:
        to_csv(results, output_filename)
