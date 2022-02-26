#!/usr/bin/env python
"""
Scrape FishBase information and Save as pickle

How to use:

    Fetch information for "Abalistes stellatus", and saves to "data/Abalistes stellatus.pkl"
    $ echo "Abalistes stellatus" | python scraper.py

    To process many in parallel, run following instead. (Install GNU parallel beforehand)
    $ cat list.txt | parallel --keep-order echo {} | python scraper.py" > list.log

"""

import bs4
import urllib.parse
import urllib.request
import urllib.error
import pickle
import pathlib


URL = "https://fishbase.se/summary/"


def request(genus_species: str) -> str:
    """Get a HTML page corresponding to `genus_species` input.
    [NOTE] genus_species should be like "Naso hexacanthus", space-separated two words.

    Reference: https://docs.python.org/3/howto/urllib2.html
    """
    pair = genus_species.strip().split()
    name = "-".join(pair)
    req = URL + name + ".html"
    with urllib.request.urlopen(req) as response:
        page = response.read()
    return page


def scrape(html_content: str) -> dict[str, str]:
    """Scrape information fro HTML and returns them in dict.

    Dictionary has three keys:
        "Common Name", "Environment", "IUCN Red List Status"
    """
    soup = bs4.BeautifulSoup(html_content, features="html.parser")
    name = soup.find(id="ss-sciname").find(class_="sheader2").text.strip()
    environment = soup.find(id="ss-main").find_all(class_="smallSpace")[1].span.text.strip().replace(u'\xa0', ' ')
    iucn = soup.find(class_="sleft sonehalf").span.a.text.strip()

    result = {
        "Common Name": name,
        "Environment": environment,
        "IUCN Red List Status": iucn,
    }

    return result


if __name__ == "__main__":
    pathlib.Path("./data").mkdir(exist_ok=True)
    name = input()

    try:
        page = request(name)
        d = scrape(page)
        assert d
        print(f"{name}\tOK")
        path = pathlib.Path(f"./data/{name}.pkl")
        with open(path, "wb") as fobj:
            pickle.dump(d, fobj)

    except urllib.error.HTTPError:
        print(f"{name}\tFAIL (HTTPError)")
    except AttributeError:
        print(f"{name}\tFAIL (Parse Error)")

