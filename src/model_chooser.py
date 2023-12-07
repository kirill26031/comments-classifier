import os
import pickle as pk

base_dir = "models/"


def list_folders(directory_path):
    try:
        all_items = os.listdir(directory_path)

        folders = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]

        return folders

    except FileNotFoundError:
        print(f"Error: The specified directory '{directory_path}' does not exist.")
        return []


class Model:
    def __init__(self, ipca, gnbs, vectorizer):
        self.ipca = ipca
        self.gnbs = gnbs
        self.vectorizer = vectorizer


MODELS = list_folders(base_dir)
DEFAULT_MODEL = "3000-100-components"


def get_model(model_name):
    if MODELS.__contains__(model_name):
        directory = base_dir + model_name
        ipca = pk.load(open(directory + "/pca.pkl", 'rb'))
        vectorizer = pk.load(open(directory + "/vectorizer.pkl", 'rb'))
        gnbs = dict()
        gnbs_list = list_folders(directory)
        for gnb_filename in gnbs_list:
            gnb = pk.load(open(directory + "/" + gnb_filename + "/gnb.pkl", 'rb'))
            gnbs.__setitem__(gnb_filename, gnb)
        return Model(ipca, gnbs, vectorizer)
    else:
        raise ValueError("Incorrect model name")


def store_model(model_name, model):
    while MODELS.__contains__(model_name):
        model_name += '-copy'
    model_directory = base_dir + model_name
    pca_path = model_directory + "/pca.pkl"
    vectorizer_path = model_directory + "/vectorizer.pkl"
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)
    pk.dump(model.ipca, open(pca_path, "xb"))
    pk.dump(model.vectorizer, open(vectorizer_path, "xb"))
    for target, gnb in model.gnbs.items():
        target_path = model_directory + "/" + target
        if not os.path.exists(target_path):
            os.mkdir(target_path)
            pk.dump(gnb, open(target_path + "/gnb.pkl", "xb"))

