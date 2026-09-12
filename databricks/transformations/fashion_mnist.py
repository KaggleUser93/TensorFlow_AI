"""Declarative pipeline: raw Fashion-MNIST images -> features -> predictions.

Bronze and silver are pure transformations; the predictions table applies the MLlib
model trained by ../train_mlp_model.py, since model fitting cannot be expressed
declaratively.

Pipeline query functions may not trigger DataFrame analysis, so `model.transform`
cannot be called inside one. The trained network's weights are unpacked once at
graph construction and applied through a pandas UDF instead.
"""

import numpy as np
import pandas as pd
from pyspark import pipelines as dp
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.active()

RAW_TABLE = spark.conf.get("fashion.raw_table")
MODEL_PATH = spark.conf.get("fashion.model_path")

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def unpack_layers(model):
    """Split the flat MLlib weight vector into (weights, bias) per affine layer."""
    flat = model.weights.toArray()
    layers = model.getLayers()
    offset = 0
    unpacked = []
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        matrix = flat[offset:offset + n_in * n_out].reshape((n_out, n_in), order='F')
        offset += n_in * n_out
        bias = flat[offset:offset + n_out]
        offset += n_out
        unpacked.append((matrix, bias))
    return unpacked


LAYERS = unpack_layers(MultilayerPerceptronClassificationModel.load(MODEL_PATH))


@F.pandas_udf('double')
def predict(features: pd.Series) -> pd.Series:
    """Forward pass of the MLlib network: sigmoid hidden layers, argmax output."""
    activations = np.vstack(features.to_numpy())
    for index, (matrix, bias) in enumerate(LAYERS):
        activations = activations @ matrix.T + bias
        if index < len(LAYERS) - 1:
            activations = 1.0 / (1.0 + np.exp(-activations))
    return pd.Series(activations.argmax(axis=1).astype('float64'))


def class_name(column):
    names = F.array(*[F.lit(name) for name in CLASS_NAMES])
    return F.element_at(names, column.cast('int') + 1)


@dp.materialized_view(
    name='fashion_mnist_bronze',
    comment='Raw 28x28 grayscale images as flat arrays of 784 pixel values.',
)
@dp.expect_all_or_drop({
    'complete_image': 'size(pixels) = 784',
    'known_label': 'label between 0 and 9',
    'known_split': "split in ('train', 'test')",
})
def fashion_mnist_bronze():
    return spark.read.table(RAW_TABLE)


@dp.materialized_view(
    name='fashion_mnist_silver',
    comment='One feature vector per image, scaled to 0-1, ready for MLlib.',
)
def fashion_mnist_silver():
    return (
        spark.read.table('fashion_mnist_bronze')
        .select(
            'image_id',
            'split',
            F.col('label').cast('double').alias('label'),
            class_name(F.col('label')).alias('label_name'),
            array_to_vector(
                F.transform('pixels', lambda pixel: pixel.cast('double') / 255.0)
            ).alias('features'),
        )
    )


@dp.materialized_view(
    name='fashion_mnist_predictions',
    comment='Model predictions on the held-out test split.',
)
def fashion_mnist_predictions():
    return (
        spark.read.table('fashion_mnist_silver')
        .where(F.col('split') == 'test')
        .withColumn('prediction', predict(vector_to_array('features')))
        .select(
            'image_id',
            'label',
            'label_name',
            'prediction',
            class_name(F.col('prediction')).alias('predicted_name'),
            (F.col('prediction') == F.col('label')).alias('correct'),
        )
    )


@dp.materialized_view(
    name='fashion_mnist_accuracy_by_class',
    comment='Test accuracy per garment class.',
)
def fashion_mnist_accuracy_by_class():
    return (
        spark.read.table('fashion_mnist_predictions')
        .groupBy('label', 'label_name')
        .agg(
            F.count('*').alias('n'),
            F.avg(F.col('correct').cast('double')).alias('accuracy'),
        )
        .orderBy('label')
    )
