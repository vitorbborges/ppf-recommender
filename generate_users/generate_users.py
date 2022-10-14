import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import glob

#gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel

#spacy
import spacy

#vis
import pyLDAvis
import pyLDAvis.gensim_models

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('database/journals.csv')