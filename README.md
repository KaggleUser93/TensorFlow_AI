# TensorFlow_AI

A collection of TensorFlow/Keras practice notebooks: image classification (MNIST,
Fashion-MNIST, Horses or Humans, Cats vs Dogs, Rock-Paper-Scissors, sign language),
text classification (IMDB), and regression (house prices, fuel efficiency).

Most notebooks were written for Google Colab in 2019 and still rely on Colab-only
features such as `google.colab` file uploads and `!pip` magics.

## Running locally

`TF_Proj2_ZalandoFashion.ipynb` runs as-is on current TensorFlow without Colab:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook TF_Proj2_ZalandoFashion.ipynb
```

It trains on CPU in a couple of minutes and reaches roughly 88% test accuracy.
Keras downloads the Fashion-MNIST dataset automatically on first run.

`TF_Proj2_ZalandoFashion_PySpark.ipynb` is the same model ported to Spark MLlib's
`MultilayerPerceptronClassifier`. It needs a JVM (Java 17 works), trains in about
five minutes locally, and reaches roughly 87% test accuracy. TensorFlow is used
there only to download the dataset.

## Databricks declarative pipeline

`databricks/` holds the same Fashion-MNIST workflow as a Spark Declarative
Pipeline (Lakeflow):

| File | Role |
| --- | --- |
| `ingest_raw_fashion_mnist.py` | One-off job: downloads Fashion-MNIST and writes the raw table. |
| `train_mlp_model.py` | Job: fits the 784-128-10 MLlib classifier and saves it. |
| `transformations/fashion_mnist.py` | The pipeline: bronze -> silver -> predictions -> per-class accuracy. |
| `pipeline.yml` | Pipeline spec (source glob plus the table and model paths). |

Model fitting is imperative, so it stays outside the pipeline; the pipeline loads
the saved model and applies it. Because pipeline query functions may not trigger
DataFrame analysis, `model.transform` cannot be called inside a dataset function —
the network weights are unpacked once at graph construction and applied through a
pandas UDF.

### Running it in a workspace

1. Import this folder into the workspace (Git folder or workspace files).
2. Run `ingest_raw_fashion_mnist.py` once as a notebook or job, e.g.
   `--table main.default.fashion_mnist_raw`.
3. Run `train_mlp_model.py` with `--model-path` pointing at a Unity Catalog volume.
4. Create a pipeline whose source is `databricks/transformations`, then set
   `fashion.raw_table` and `fashion.model_path` in the pipeline configuration
   (the values in `pipeline.yml` are the defaults to copy).
5. Trigger an update; it materializes `fashion_mnist_bronze`, `_silver`,
   `_predictions`, and `_accuracy_by_class`.

### Running it locally

The pipeline also runs against open-source Spark, which is how it was verified:

```bash
cd databricks
python ingest_raw_fashion_mnist.py --table fashion_mnist_raw --limit-per-split 3000
python train_mlp_model.py --table fashion_mnist_raw --model-path /tmp/fashion_mnist_model --max-iter 20
spark-pipelines run --spec pipeline.yml
```

Point `fashion.raw_table` and `fashion.model_path` in `pipeline.yml` at those
local values first. The `spark-pipelines` CLI needs the Spark Connect extras,
which `requirements.txt` installs.

Open-source Spark has no expectations support, so the `@dp.expect_all_or_drop`
decorator on the bronze table has to be commented out for a local run. Everything
else is identical.
