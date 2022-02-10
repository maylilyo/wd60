# Standard
from pathlib import Path
import time

# Custom
from dataset import Vimeo


def benchmark():
    project_dir = Path(__file__).absolute().parent.parent.parent
    data_dir = project_dir / "data"

    test_count = 1000
    state = "train"

    start_time = time.time()
    test_dataset = Vimeo(data_dir, state, is_pt=True)
    init_time = time.time()
    print(f"Init: {init_time - start_time}")
    if len(test_dataset) < test_count:
        test_count = len(test_dataset)
    print(f"Count: {test_count}")
    init_time = time.time()
    for idx, data in enumerate(test_dataset):
        if (idx + 1) % test_count == 0:
            break
    read_time = time.time() - init_time
    print(f"Read: {read_time}")
    print(f"Avg : {read_time / test_count}")


if __name__ == "__main__":
    benchmark()
