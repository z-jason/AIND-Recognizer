import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import asl_data


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except Exception:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []
        for n in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(n)

            if model is None:
                continue

            try:
                log_l = model.score(self.X, self.lengths)
            except ValueError:
                # NOTE: Only happen to 'FISH' when 6 <= n <= 15
                #   This issue is reported in this post: http://bit.ly/2xfola5
                continue

            score = self._get_bic_score(log_l, n)
            models.append((score, model))

        if len(models) == 0:
            return None

        return sorted(models)[0][1]

    # The smaller BIC score the better
    def _get_bic_score(self, log_l, num_states):
        """

        Parameters
        ----------
        log_l
        num_states

        Returns
        -------

        References
        ----------
        bic = -2 * logL + p * logN
        http://bit.ly/2xfQKgr
        http://bit.ly/2xgwn2C
        https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
        """
        num_of_features = len(self.X[0])
        parameter = num_states ** 2 + 2 * num_states * num_of_features - 1
        log_n = math.log(len(self.X))
        return -2 * log_l + parameter * log_n


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []
        for n in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(n)
            if model is None:
                continue
            try:
                log_l = model.score(self.X, self.lengths)
                anti_log_ls = [model.score(x, lengths) for x, lengths in self._other_word_Xlengths()]
            except ValueError:
                continue
            score = log_l - np.mean(anti_log_ls)
            models.append((score, model))

        if len(models) == 0:
            return None

        return sorted(models)[-1][1]

    def _other_word_Xlengths(self):
        for word, (X, lengths) in self.hwords.items():
            if word != self.this_word:
                yield X, lengths


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # If the given word has less than 3 sample, it does not make sense to do the K-fold CV
        # Just return something the same as SelectorConstant
        if len(self.sequences) < 3:
            return self.base_model(self.n_constant)

        best_num_states = self._get_best_num_states(self._score_func)

        return self.base_model(best_num_states)

    def _score_func(self, num_states):
        model = GaussianHMM(n_components=num_states,
                            covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False)
        logLs = []
        split_method = KFold(n_splits=3)
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))
            training_X, training_lengths = combine_sequences(cv_train_idx, self.sequences)
            testing_X, testing_lengths = combine_sequences(cv_test_idx, self.sequences)
            trained_model = model.fit(training_X, training_lengths)
            try:
                logL = trained_model.score(testing_X, testing_lengths)
            # Saw this when training w/ word 'VEGETABLE' when num_states = 13
            #   `ValueError: rows of transmat_must sum to 1.0 (got [ 1.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.])`
            # NOTE: Potential this statement may cause `logLs` to be empty
            except ValueError:
                continue
            logLs.append(logL)
        try:
            return sum(logLs) / len(logLs)
        except ZeroDivisionError:
            return float('-inf')

    def _get_best_num_states(self, score_func):
        scores = self._get_scores(score_func)
        _, best_num_states = sorted(scores, reverse=True)[0]
        return best_num_states

    def _get_scores(self, score_func):
        return [(score_func(n), n) for n in range(self.min_n_components, self.max_n_components+1)]

