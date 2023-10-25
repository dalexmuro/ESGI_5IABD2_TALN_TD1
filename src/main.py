import click
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
import pandas as pd
from src.data.make_dataset import make_dataset
from src.features.make_features import make_features
from src.model.main import make_model
import joblib
import pickle
import gzip


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/model.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model()
    model.fit(X, y)

    with gzip.open(model_dump_filename, 'wb') as f:
        pickle.dump(model, f)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    pass



@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/test.csv", help="File training data")
@click.option("--model_dump_filename", default="models/text_classification.pkl", help="File to load model from")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    X = make_features(df, task, output_required=False)
    model = pickle.load(gzip.open(model_dump_filename, 'rb'))
    df[task] = model.predict(X)

    return df.to_csv(output_filename, index=False)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    results = cross_validate(model, X, y, scoring="accuracy", cv=10, return_estimator=True)
    scores = results["test_score"]


    print(f"Got accuracy {100 * np.mean(scores)}%")
    print(scores)

    vocab = results['estimator'][0]['count_vectorizer'].vocabulary_
    keys = list(vocab.keys())


    importance = results['estimator'][0]['tree'].feature_importances_
    features = pd.DataFrame(importance, index=keys, columns=['importance']).sort_values('importance', ascending=False)

    print(features.head(10))

    return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()