from glob import glob
import os
import pandas as pd
import pickle

HOME_DIR = '/'.join(os.getcwd().split('/')[:-1])
DB_DIR = os.path.join(HOME_DIR, ".database")

PIC_EXTS = "jpg", "JPG", "jpeg", "JPEG", "png", "PNG", "gif", "GIF"
EMB_EXTS = "pkl",


def save_embedding_to_db(person_name, file_name, embedding, database_dir=DB_DIR):
    if not os.path.exists(os.path.join(database_dir, person_name)):
        os.mkdir(os.path.join(database_dir, person_name))
    
    with open(os.path.join(database_dir, person_name, file_name), "wb") as fp:
        pickle.dump(embedding, fp) # save numpy array as pickle


def load_pickle_emb_as_np_arr(pickle_full_path):
    with open(pickle_full_path, "rb") as fp:
        numpy_array_embedding = pickle.load(fp)
    return numpy_array_embedding


def load_pictures_to_df(db_dir=DB_DIR):
    # 파일명
    file_paths = [ fname for fname in glob(os.path.join(db_dir,"**"), recursive=True) if fname.split('.')[-1] in PIC_EXTS ]

    # 폴더명 (= 사람 이름)
    folder_names = [ folder_name.replace(db_dir, '') for folder_name in file_paths ]
    folder_names = [ folder_name[1:] if folder_name[0]=='/' else folder_name for folder_name in folder_names ]
    folder_names = [ folder_name.split('/')[0] for folder_name in folder_names ]

    df = pd.DataFrame.from_dict({
        "name": folder_names,
        "pic_path": file_paths
    })
    
    return df


def load_embeddings_to_df(db_dir=DB_DIR):
    # 파일명
    file_paths = [ fname for fname in glob(os.path.join(db_dir,"**"), recursive=True) if fname.split('.')[-1] in EMB_EXTS ]

    # 임베딩 벡터
    numpy_arrays = [ load_pickle_emb_as_np_arr(fname) for fname in file_paths ]

    # 폴더명 (= 사람 이름)
    folder_names = [ folder_name.replace(db_dir, '') for folder_name in file_paths ]
    folder_names = [ folder_name[1:] if folder_name[0]=='/' else folder_name for folder_name in folder_names ]
    folder_names = [ folder_name.split('/')[0] for folder_name in folder_names ]

    df = pd.DataFrame.from_dict({
        "name": folder_names,
        "emb_path": file_paths,
        "emb_arrs": numpy_arrays
    })
    
    return df
