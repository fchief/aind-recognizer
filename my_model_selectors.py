import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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
        except:
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

        # TODO implement model selection based on BIC scores
        best_bic = float("Inf")
        best_model = None
        for ns in range(self.min_n_components, self.max_n_components+1):
            try:
                temp_model = self.base_model(ns)
                temp_logL = temp_model.score(self.X, self.lengths)
                N = len(self.X)
                p = ns*(ns-1)+(ns-1)+2*self.X.shape[1]*ns
                bic = -2*temp_logL + p*np.log(N)
                if bic < best_bic:
                    best_bic = bic
                    best_model = temp_model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    # Calculate anti log likelihoods.
    def calc_log_likelihood_other_words(self, model, other_words):
        return [model[1].score(word[0], word[1]) for word in other_words]

    def calc_best_score_dic(self, score_dics):
        # Max of list of lists comparing each item by value at index 0
        return max(score_dics, key = lambda x: x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_dic = float("-inf")
        best_model = None
        try:
            for ns in range(self.min_n_components, self.max_n_components + 1):
                temp_model = self.base_model(ns)
                temp_logL = temp_model.score(self.X, self.lengths)
                temp_meanLogL = np.mean([temp_model.score(word[0], word[1]) for word in self.hwords[word] if word != self.this_word])
                dic = temp_logL - temp_meanLogL
                if best_dic < dic:
                    best_dic = dic
                    best_model = temp_model
        except:
            pass
        return best_model if best_model else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float("-inf")
        best_model = None
        for ns in range(self.min_n_components, self.max_n_components + 1):
            if len(self.sequences) <= 1:
                continue

            split_method = KFold(n_splits=min(3, len(self.sequences)))

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                logL_scores = []
                self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                X_test, length_test = combine_sequences(cv_test_idx, self.sequences)
                try:
                    temp_model = self.base_model(ns)
                    temp_CV = np.mean(temp_model.score(X_test, length_test))
                    if temp_CV > best_score:
                        best_score = temp_CV
                        best_model = temp_model
                except:
                    pass

        return best_model if best_model else None
