from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline

def make_model(task):
    if task == "is_comic_video":
        return Pipeline([
            ("count_vectorizer", CountVectorizer(min_df=2, lowercase=False)),
            ("tree", GradientBoostingClassifier(n_estimators=150, criterion='squared_error')),
        ])
    elif task == "is_name":
        return Pipeline([
            ("count_vectorizer", CountVectorizer(min_df=2, lowercase=False)),

        ])