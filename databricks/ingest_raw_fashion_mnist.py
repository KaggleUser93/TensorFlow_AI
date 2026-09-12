"""Write the Fashion-MNIST images into the raw table the pipeline reads.

Run once before the pipeline (as a Databricks notebook or job task):

    %pip install tensorflow-cpu
    %run ./ingest_raw_fashion_mnist

Locally:

    python ingest_raw_fashion_mnist.py --table main.default.fashion_mnist_raw
"""

import argparse

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (ArrayType, IntegerType, LongType, StringType,
                               StructField, StructType)
from tensorflow import keras

SCHEMA = StructType([
    StructField('image_id', LongType()),
    StructField('split', StringType()),
    StructField('label', IntegerType()),
    StructField('pixels', ArrayType(IntegerType())),
])


def raw_dataframe(spark, limit_per_split=None):
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    frames = []
    for split, images, labels in (('train', train_images, train_labels),
                                  ('test', test_images, test_labels)):
        if limit_per_split:
            images, labels = images[:limit_per_split], labels[:limit_per_split]
        flat = images.reshape(len(images), -1)
        frames.append(pd.DataFrame({
            'image_id': range(len(flat)),
            'split': split,
            'label': labels.astype(int),
            'pixels': [row.tolist() for row in flat],
        }))

    pdf = pd.concat(frames, ignore_index=True)
    pdf['image_id'] = range(len(pdf))
    return spark.createDataFrame(pdf, schema=SCHEMA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', default='main.default.fashion_mnist_raw')
    parser.add_argument('--limit-per-split', type=int, default=None,
                        help='Ingest only the first N images of each split.')
    args = parser.parse_args()

    spark = SparkSession.builder.appName('fashion-mnist-ingest').getOrCreate()
    df = raw_dataframe(spark, args.limit_per_split)
    df.write.mode('overwrite').saveAsTable(args.table)
    print(f'wrote {df.count()} rows to {args.table}')


if __name__ == '__main__':
    main()
