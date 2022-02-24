# Scraping FishBase

## About

The script scrapes [FishBase](https://fishbase.se/search.php) pages and extracts following items.

1. Common Name
2. Environment
3. IUCN Red List Status


## Requirements

* beautifulsoup4
* pathos [optional]

If you are `conda` user, it's recommended to create a virtual environment with

```shell
$ conda create -n fishbase -c conda-forge pathos beautifulsoup4
$ conda activate fishbase
```


## Using the scripts

### 0. Clone this repository

⚠️ Please install `git-lfs`⚠️  before cloning the repository as this it contains a binary file (gzip) managed by Git LFS. 

```shell
# Ubuntu
$ sudo apt install git-lfs

# MacOS
$ brew install git-lfs
```

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
$ python utils/merge_and_keep_unique_names.py file1 file2 > list.txt
```


### 0-b. Check names in the list

It might be good to check if names in your list exist in FishBase beforehand. Following command will show (a) number of matches and (b) suggested corrections.

```shell
$ python check_names.py list.txt
```

**[Optional]** If you want to save the corrections as CSV, a pipe like this might be useful; the first column has the original names, and the second column has suggested names. Failed suggestions are either `[OK by words; not found in FishBase]` or `?????`.

```shell
$ python check_names.py list.txt > suggestions.txt
$ cat suggestions.txt | sed -r 's/\t-->\t/,/g' > suggestions.csv
```


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
$ cat list.txt | parallel "echo {} | python scraper.py" | sort > list.log
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
