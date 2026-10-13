import zipfile
from squad_preprocess import *


if __name__ == '__main__':
    glove_base_url = "https://nlp.stanford.edu/data/"
    glove_filename = "glove.840B.300d.zip"
    prefix = os.path.join("data", "dwr")

    print("Storing datasets in {}".format(prefix))

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    glove_zip = maybe_download(glove_base_url, glove_filename, prefix)
    glove_zip_ref = zipfile.ZipFile(os.path.join(prefix, glove_filename), 'r')

    glove_zip_ref.extractall(prefix)
    glove_zip_ref.close()
