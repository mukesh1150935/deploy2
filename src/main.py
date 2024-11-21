from  data.split_data import run as run_split
from data_pre import run as run_pre
from data.count import run as run_count


if __name__ == "__main__":
    run_split()
    run_pre()
    run_count()