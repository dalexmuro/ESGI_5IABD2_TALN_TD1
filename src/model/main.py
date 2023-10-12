from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline

def make_model():
    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        ("tree", GradientBoostingClassifier(n_estimators=150, criterion='squared_error')),
    ])