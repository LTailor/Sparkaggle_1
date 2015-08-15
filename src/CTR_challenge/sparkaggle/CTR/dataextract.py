__author__ = 'Richard Walker, Dzhambulat Khasayev'

import numpy as np
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from collections import defaultdict
import hashlib


def getHashData(rawData,numBucketsCTR):
    """Conver raw data to hashed data

    Args:
        rawData: distributed list of strings. Each string contains a label (0.0 or 1.0) and features. These all are comma separated
        numBuckets: The number of buckets to hash to.

    Returns:
        collection of LabeledPoints with a same label and a SparseVetor of hashed features.
    """

    hashData = (rawData
    .map(lambda point: parseHashPoint(point, numBucketsCTR)))

    return hashData

def parseHashPoint(point, numBuckets):
    """Create a LabeledPoint for this observation using hashing.

    Args:
        point (str): A comma separated string where the first value is the label and the rest are
            features.
        numBuckets: The number of buckets to hash to.

    Returns:
        LabeledPoint: A LabeledPoint with a label (0.0 or 1.0) and a SparseVector of hashed
            features.
    """
    splits = [s for s in point.split(',')]
    label = splits.pop(0)
    parsedPoint = [(i, splits[i]) for i in range(len(splits))]
    # parsedPoint is list of tuples of indexed values. The values are either 0 or 1. The index range is large; for example, here they go up to 32768 with the param passed in for the exercise.

    # hashFunction returns dict of int to float, i.e. the keys will be integers which represent the buckets that the features have been hashed to. The value for a given key will contain the count of the (featureID, value) tuples that have been hashed to that key.

    hashDict = hashFunction(numBuckets, parsedPoint, printMapping=False)
    lp = LabeledPoint(label, SparseVector(numBuckets, hashDict))
    return lp

def hashFunction(numBuckets, rawFeats, printMapping=False):
    """Calculate a feature dictionary for an observation's features based on hashing.

    Note:
        Use printMapping=True for debug purposes and to better understand how the hashing works.

    Args:
        numBuckets (int): Number of buckets to use as features.
        rawFeats (list of (int, str)): A list of features for an observation.  Represented as
            (featureID, value) tuples.
        printMapping (bool, optional): If true, the mappings of featureString to index will be
            printed.

    Returns:
        dict of int to float:  The keys will be integers which represent the buckets that the
            features have been hashed to.  The value for a given key will contain the count of the
            (featureID, value) tuples that have hashed to that key.
    """
    mapping = {}
    for ind, category in rawFeats:
        featureString = category + str(ind)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % numBuckets)
    if(printMapping): print mapping
    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)