"""Fit the 784 -> 128 -> 10 MLlib classifier and save it for the pipeline.

Model fitting is imperative, so it lives outside the declarative pipeline. Rerun
this job whenever the training data changes; the pipeline picks up the new model
on its next update.

    python train_mlp_model.py --table main.default.fashion_mnist_raw \
        --model-path /Volumes/main/default/models/fashion_mnist_mlp
"""

import argparse

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import array_to_vector
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def features(df):
    return df.select(
        F.col('label').cast('double').alias('label'),
        array_to_vector(
            F.transform('pixels', lambda pixel: pixel.cast('double') / 255.0)
        ).alias('features'),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', default='main.default.fashion_mnist_raw')
    parser.add_argument('--model-path', default='/Volumes/main/default/models/fashion_mnist_mlp')
    parser.add_argument('--max-iter', type=int, default=100)
    args = parser.parse_args()

    spark = SparkSession.builder.appName('fashion-mnist-train').getOrCreate()
    raw = spark.read.table(args.table)

    train = features(raw.where(F.col('split') == 'train')).cache()
    test = features(raw.where(F.col('split') == 'test')).cache()

    model = MultilayerPerceptronClassifier(
        layers=[784, 128, 10],
        blockSize=128,
        maxIter=args.max_iter,
        seed=42,
    ).fit(train)

    accuracy = MulticlassClassificationEvaluator(metricName='accuracy').evaluate(
        model.transform(test)
    )
    print('test accuracy', accuracy)

    model.write().overwrite().save(args.model_path)
    print('saved model to', args.model_path)


if __name__ == '__main__':
    main()
