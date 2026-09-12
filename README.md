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
