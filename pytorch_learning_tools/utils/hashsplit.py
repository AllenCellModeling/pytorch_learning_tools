import hashlib
import numpy as np

def hashsplit(X, splits={'train': 0.8, 'test': 0.2}, salt=1, N=5):
    """
    splits a list of items pseudorandomly (but deterministically) based on the hashes of the items
    Args:
        X (list): list of items to be split into non-overlapping groups
        splits = (dict): dict of {name:weight} pairs definiting the desired split
        salt (string): str(salt) is appended to each list item before hashing
        N (int): number of significant figures to compute for binning each list item
    Returns:
        dict of {name:indices} for all names in the input split dict
    Example:
        >>> hashsplit(list("allen cell institute"), {'train':0.7,'test':0.3}, salt=3, N=8) 
        {'test': [4, 12, 17],
        'train': [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]}
    """

    # normalize the weights, just in case
    splits = {k: v / sum(splits.values()) for k, v in splits.items()}

    # determine bins in [0,1] that correspond to each split
    bounds = np.cumsum([0.0] + [v for k, v in sorted(splits.items())])
    bins = {k: [bounds[i], bounds[i + 1]] for i, (k, v) in enumerate(sorted(splits.items()))}

    # hash the strings deterministically
    hashes = [hashlib.sha512((str(x) + str(salt)).encode('utf-8')).hexdigest() for x in X]

    # create some numbers in [0,1] (at N sig figs) from the hashes
    nums = np.array([float("".join(filter(str.isdigit, h))[:N]) / 10**N for h in hashes])

    # check where the nums fall in [0,1] relative to the bins left and right boundaries
    inds = {k: np.where((nums > l) & (nums <= r)) for k, (l, r) in bins.items()}

    # np.where returns a singleton tuple containing an np array, so convert to list
    return {k: list(*v) for k, v in inds.items()}

