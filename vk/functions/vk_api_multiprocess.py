from multiprocessing import Pool
from tqdm import tqdm_notebook
import numpy as np

from functions.vk_api import *


def MultiGroupsSearch(texts, count=1000, verbose=False, processes=2):
    '''
    Не совсем работает, приходится самому прописывать user_token
    '''
    iterator = tqdm_notebook([(i, count) for i in texts]) if verbose else [(i, count) for i in texts]
    with Pool(processes=processes) as p:
        return np.hstack(p.starmap(GroupsSearch, iterator))
    