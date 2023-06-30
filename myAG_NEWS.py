import os
from functools import partial
from typing import Union, Tuple

from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._download_hooks import HttpReader
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

URL = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}

MD5 = {
    "train": "b1a00f826fdfbd249f79597b59e1dc12",
    "test": "d52ea96a97a2d943681189a97654912d",
}

NUM_LINES = {
    "train": 120000,
    "test": 7600,
}

DATASET_NAME = "AG_NEWS"


def _filepath_fn(root, split, _=None):
    return os.path.join(root, split + ".csv")


def _modify_res(t):
    label, text = int(t[0]), " ".join(t[1:])

    if label in [1, 2]:
        return label, text
    else:
        return None  # Skip other labels


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def AG_NEWS(root: str, split: Union[Tuple[str], str]):
    """AG_NEWS Dataset
    ...
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    url_dp = IterableWrapper([URL[split]])
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root, split),
        hash_dict={_filepath_fn(root, split): MD5[split]},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp)
    cache_dp = cache_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_dp, encoding="utf-8")
    return (
        data_dp.parse_csv()
        .map(fn=_modify_res)
        .filter(filter_fn=lambda x: x is not None)  # Skip None values
        .shuffle()
        .set_shuffle(False)
        .sharding_filter()
    )
