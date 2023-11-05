import string
import itertools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import pandas as pd
from nltk.stem.snowball import FrenchStemmer
from flair.data import Sentence
from flair.models import SequenceTagger

stop_words = ["a","à","â","abord","afin","ah","ai","aie","ainsi","allaient","allo","allô","allons","après","assez","attendu","au","aucun","aucune","aujourd","aujourd'hui","auquel","aura","auront","aussi","autre","autres","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avoir","ayant","b","bah","beaucoup","bien","bigre","boum","bravo","brrr","c","ça","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chaque","cher","chère","chères","chers","chez","chiche","chut","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","delà","depuis","derrière","des","dès","désormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","différent","différente","différentes","différents","dire","divers","diverse","diverses","dix","dix-huit","dixième","dix-neuf","dix-sept","doit","doivent","donc","dont","douze","douzième","dring","du","duquel","durant","e","effet","eh","elle","elle-même","elles","elles-mêmes","en","encore","entre","envers","environ","es","ès","est","et","etant","étaient","étais","était","étant","etc","été","etre","être","eu","euh","eux","eux-mêmes","excepté","f","façon","fais","faisaient","faisant","fait","feront","fi","flac","floc","font","g","gens","h","ha","hé","hein","hélas","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","i","il","ils","importe","j","je","jusqu","jusque","k","l","la","là","laquelle","las","le","lequel","les","lès","lesquelles","lesquels","leur","leurs","longtemps","lorsque","lui","lui-même","m","ma","maint","mais","malgré","me","même","mêmes","merci","mes","mien","mienne","miennes","miens","mille","mince","moi","moi-même","moins","mon","moyennant","n","na","ne","néanmoins","neuf","neuvième","ni","nombreuses","nombreux","non","nos","notre","nôtre","nôtres","nous","nous-mêmes","nul","o","o|","ô","oh","ohé","olé","ollé","on","ont","onze","onzième","ore","ou","où","ouf","ouias","oust","ouste","outre","p","paf","pan","par","parmi","partant","particulier","particulière","particulièrement","pas","passé","pendant","personne","peu","peut","peuvent","peux","pff","pfft","pfut","pif","plein","plouf","plus","plusieurs","plutôt","pouah","pour","pourquoi","premier","première","premièrement","près","proche","psitt","puisque","q","qu","quand","quant","quanta","quant-à-soi","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelque","quelques","quelqu'un","quels","qui","quiconque","quinze","quoi","quoique","r","revoici","revoilà","rien","s","sa","sacrebleu","sans","sapristi","sauf","se","seize","selon","sept","septième","sera","seront","ses","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soit","soixante","son","sont","sous","stop","suis","suivant","sur","surtout","t","ta","tac","tant","te","té","tel","telle","tellement","telles","tels","tenant","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutes","treize","trente","très","trois","troisième","troisièmement","trop","tsoin","tsouin","tu","u","un","une","unes","uns","v","va","vais","vas","vé","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voilà","vont","vos","votre","vôtre","vôtres","vous","vous-mêmes","vu","w","x","y","z","zut","alors","aucuns","bon","devrait","dos","droite","début","essai","faites","fois","force","haut","ici","juste","maintenant","mine","mot","nommés","nouveaux","parce","parole","personnes","pièce","plupart","seulement","soyez","sujet","tandis","valeur","voie","voient","état","étions"]

def remove_stopwords(sentence):
    sentence = [w for w in sentence if not w.lower() in stop_words]
    return sentence


def stemming(sentence):
    stemmer = FrenchStemmer()
    return [stemmer.stem(X) for X in sentence]


def remove_numbers(sentence):
    filtered_sentence = [w for w in sentence if not w.isnumeric()]
    return filtered_sentence


def remove_punct(sentence):
    punct = '!"#$%&()*+,./:;<=>?@[\]^_{|}~'
    sentence = sentence.translate(str.maketrans("", "", punct))
    return sentence


def tokenize(sentence):
    sentence = remove_punct(sentence)
    sentence = sentence.replace("'", "' ").replace("’", "’ ").split()
    return sentence


def tagging(sentence):
    new_sentence = []
    for w in sentence:
        # First word : <s> (starting)
        word = w + "<s>" if sentence.index(w) == 0 else w
        # Last word : <e> (end)
        word = word + "<e>" if sentence.index(w) == len(sentence)-1 else word
        # Capitalized word : <c> (caps)
        word = word.lower() + "<c>" if w[0].isupper() else word

        new_sentence.append(word)

    return new_sentence


def pos_tagging(tagger, phrase, is_name):
    result = pd.DataFrame(columns=['words', 'tags'])
    for i in range(len(phrase)):
        word = phrase[i]

        if word:
            sentence = Sentence(word)
            tagger.predict(sentence)

            # First word : <s> (starting)
            word = word + "<s>" if i == 0 else word
            # Last word : <e> (end)
            word = word + "<e>" if i == (len(phrase) - 1) else word
            # Capitalized word : <c> (caps)
            word = word + "<c>" if len(word.replace('"', '').strip()) > 0 and (word.replace('"', '').strip())[
                0].isupper() else word

            values = pd.DataFrame([{'words': word, 'tags': sentence.tag}])

            # else:
            #    values = pd.DataFrame([{'words': word, 'tags': ''}])

            result = pd.concat([result, values], axis=0, ignore_index=True)

    result['is_name'] = pd.DataFrame({'is_name': is_name})

    return result


def make_features(df, task, output_required=True):
    if task == 'is_comic_video':
        X = df["video_name"].apply(tokenize)
        X = [remove_stopwords(i) for i in X]
        X = [tagging(i) for i in X]
        X = [' '.join(i) for i in X]

        y = get_output(df, task)

    elif task == 'is_name':
        tagger = SequenceTagger.load("qanastek/pos-french")

        df["is_name"] = df["is_name"].apply(eval)
        df["tokens"] = df["tokens"].apply(eval)

        new_df = pd.DataFrame(columns=['words', 'tags', 'is_name'])

        for i in df.index:
            tokens = df.loc[i, 'tokens']
            names = [int(i) for i in df.loc[i, 'is_name']]
            result = pos_tagging(tagger, tokens, names)
            new_df = pd.concat([new_df, result], axis=0, ignore_index=True)

        X = new_df.drop('is_name', axis=1).to_dict('records')
        y = get_output(new_df, task)

    if output_required:
        return X, y
    else:
        return X


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"].values.astype(int)
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y