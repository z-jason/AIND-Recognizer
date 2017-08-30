import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    for word_id, (X, lengths) in test_set.get_all_Xlengths().items():
        prob = {}
        for word, model in models.items():
            if model is None:
                continue

            try:
                score = model.score(X, lengths)
                prob[word] = score
            except ValueError:
                # The models of these words always raise an ValueError
                #   ANN, CANDY, FIND, LEG, OLD, SAY-1P
                continue
        probabilities.append(prob)

    guesses = []
    for prob in probabilities:
        assert len(prob) > 0
        _, guess = max([(v, k) for k, v in prob.items()])
        guesses.append(guess)

    return probabilities, guesses
