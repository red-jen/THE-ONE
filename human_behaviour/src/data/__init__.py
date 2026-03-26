# Data module for Market-1501 dataset handling
from .market1501_dataset import (
    ImageInfo,
    parse_filename,
    load_dataset_split,
    get_dataset_statistics,
    create_identity_mapping,
    group_by_identity,
    group_by_camera,
    split_query_gallery,
    compute_distance_matrix,
    evaluate_reid,
)

# PyTorch datasets (only available if torch is installed)
try:
    from .market1501_dataset import Market1501Dataset, TripletMarket1501
except ImportError:
    pass

__all__ = [
    'ImageInfo',
    'parse_filename', 
    'load_dataset_split',
    'get_dataset_statistics',
    'create_identity_mapping',
    'group_by_identity',
    'group_by_camera',
    'split_query_gallery',
    'compute_distance_matrix',
    'evaluate_reid',
    'Market1501Dataset',
    'TripletMarket1501',
]
