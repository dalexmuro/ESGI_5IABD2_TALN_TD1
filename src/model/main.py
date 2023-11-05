from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

def make_model(task):
    """ USING GRID SEARCH

    pipeline = Pipeline([])
    params = []
    if task == "is_comic_video":
        pipeline = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("tree", GradientBoostingClassifier()),
        ])
        params = [{
            'vectorizer__min_df': [1, 2],
            'vectorizer__lowercase': [False],
            'tree__n_estimators': [100, 150, 200],
            'tree__loss': ['log_loss', 'exponential'],
            'tree__criterion': ['friedman_mse', 'squared_error'],
            'tree__learning_rate': [0.1, 0.01, 0.001]
        }]
    elif task == "is_name":
        pipeline = Pipeline([
            ("count_vectorizer", CountVectorizer()),
            ("tree", GradientBoostingClassifier()),
        ])
        params = [{
            'vectorizer__min_df': [1, 2],
            'vectorizer__lowercase': [False],
            'tree__n_estimators': [100, 150, 200],
            'tree__loss': ['log_loss', 'exponential'],
            'tree__criterion': ['friedman_mse', 'squared_error'],
            'tree__learning_rate': [0.1, 0.01, 0.001]
        }]
    return pipeline, params
    """
    if task == "is_comic_video":
        return Pipeline([
            ("vectorizer", CountVectorizer(min_df=2, lowercase=False)),
            ("classifier", GradientBoostingClassifier(n_estimators=150, criterion='squared_error')),
        ])
    elif task == "is_name":
        return Pipeline([
            ("vectorizer", DictVectorizer(sparse=False)),
            ("classifier", GradientBoostingClassifier(n_estimators=150, criterion='squared_error')),
        ])
