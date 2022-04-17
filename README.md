# Scraping FishBase

## About

The script scrapes [FishBase](https://fishbase.se/search.php) pages and extracts following items.

1. Common Name
2. Environment
3. IUCN Red List Status


## Requirements

* beautifulsoup4
* entrez-direct

If you are `conda` user, it's recommended to create a virtual environment with

```shell
$ conda create -n fishbase -c conda-forge beautifulsoup4
$ conda install -c bioconda entrez-direct
$ conda activate fishbase
```


## Using the scripts

### 0. Clone this repository

Clone this repository and enter the directory.

```shell
$ git clone https://github.com/yamaton/fishbase-scraper.git
$ cd fishbase-scraper
```

### 0-a. Prepare a list of species

Assume you have a text file containing genus-species names like following.

```
Abalistes stellatus
Ablennes hians
Abudefduf sexfasciatus
Abudefduf sordidus
Abudefduf sparoides
```

**[Optional]** When dealing with multiple files, `utils/merge_and_keep_unique_names.py` might be helpful. This utility script merges files and removes duplicates.

```shell
$ python utils/merge_and_keep_unique_names.py file1 file2 file3 > list.txt
```


### 0-b. Check names in the list

It might be good to check if names in your list exist in FishBase beforehand. Following command will show (a) number of matches and (b) suggested corrections.

```shell
$ python check_names.py list.txt
```


To save the corrections in CSV,  you may add `--output <file.csv>`

```shell
$ python check_names.py list.txt --output check_names_result.csv
```

Note that the CSV output contains seven columns.
1. Original names in the `list.txt`
2. Status of the name correction.
    * 🐟: Found in FishBase
    * 😃: Found correction as the concensus of multiple sources
    * Empty: Multiple correction candidates exist.
    * ❓: No idea
3. Correction suggested by WoRMS
4. Correction suggested by NCBI Taxonomy (via Entrez-direct)
5. Correction suggested by the built-in spell corrector
6. Another correction suggested by the spell corrector
7. Yet another suggested by the spell corrector


**[Optional]** This name checker automatically scrapes and downloads all scientific names in FishBase for reference. If you want to create the list of all fish names in FishBase, run

```shell
$ bash utils/collect_all_names.sh > ScientificNamesAll.txt
```


### 1. Scrape FishBase pages

Load a Fishbase page and extract information. For example, following command fetches a page and saves a Python dictionary as a pickle at  `data/Abalistes stellatus.pkl`.

```shell
$ echo "Abalistes stellatus" | python scraper.py
```


Practically, we would want to run this script against multiple species **in parallel**. We recommend using [GNU parallel](https://www.gnu.org/software/parallel/) for the job. (And activate conda environment in each of parallel processes.)

```shell
$ cat list.txt | parallel --keep-order "echo {} | python scraper.py" > list.log
```

Scraping can fail for various reasons (page not found, irregular page format, server too busy, network problems, etc.) so it's good practice to check the log after running the script. The log file `list.log`  contains tab-separated two columns. The first is the same as in `list.txt`. The second column contains either `OK` or `FAIL` as the scraping status. One can view just errors by filtering with `grep`.

```shell
$ cat list.log | grep FAIL
Gerres poeti    FAIL (Parse Error)
```



### 2. Put together data as CSV

Create a CSV file (`fishbase.csv`) from a bunch of pickle files under `data` directory.

```shell
$ python to_csv.py
```

Note that the step #1 and #2 were separated because #1 is parallelizable while #2 is not.
