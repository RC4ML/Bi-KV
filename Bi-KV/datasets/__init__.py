from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .games import GamesDataset
from .books import BooksDataset
from .clothing import ClothingDataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    BooksDataset.code(): BooksDataset,
    ClothingDataset.code(): ClothingDataset,
}

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)