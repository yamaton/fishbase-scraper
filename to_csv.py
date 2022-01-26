"""
Make CSV from Python pickles

How to use:
    $ python to_csv.py
"""
import pathlib
import pickle
import csv

def format_env(env: str, count: int=3) -> str:
    """Take first three items delimited by semicolons.

    >>> format_env("baba; keke; foo; buzz; baa;)
    'baba; keke; foo'
    """
    items = env.split(";")
    first_three = items[:count]
    return ';'.join(first_three)


def write(ds: list[dict], output):
    """Write dict `ds` to CSV with name given as `output`.
    """
    with open(output, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ds[0].keys())
        writer.writeheader()
        writer.writerows(ds)


def prepare_dict_list(path: str) -> list[dict]:
    """Load a pickle files in `path` directory, and create a list of dictionary.

    Each dictionary has four keys:
        Species
        Common Name
        Environment
        IUCN Red List Status
    """
    p = pathlib.Path(path)
    files = list(p.glob("*.pkl"))
    files.sort()
    result = []
    for file in files:
        with file.open("rb") as pkl:
            print(f"  loading: {str(file)}")
            d = pickle.load(pkl)
        newdict = {
            "Species": file.stem,
            "Common Name": d["Common Name"],
            "Environment": format_env(d["Environment"]),
            "IUCN Red List Status": d["IUCN Red List Status"],
        }
        result.append(newdict)
    return result


if __name__ == "__main__":
    pickle_dir = "data"
    output = "fishbase.csv"
    ds = prepare_dict_list(pickle_dir)
    write(ds, output)
