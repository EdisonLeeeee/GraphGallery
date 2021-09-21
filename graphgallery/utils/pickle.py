import pickle


def load_pickle(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_pickle(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
    return fname
