from flair.data import Sentence
from flair.models import SequenceTagger

# Load the model
model = SequenceTagger.load("qanastek/pos-french")

sentence = Sentence("Le Barbecue Disney - La chanson de Frédéric Fromet")

# Predict tags
model.predict(sentence)

# Print predicted pos tags
print(sentence.tag)
