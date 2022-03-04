import torch
from torch.utils.data import DataLoader,Dataset
from PIL import Image

import os, cv2,itertools
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
import pickle

from arguments import get_args

# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, df ,mode, data_dir = '/opt/ml/fl_ham/input/saved_data',transform=None):
        with open(os.path.join(data_dir,mode+'.npy'), 'rb') as f:
            self.img = np.load(f)
        print(f' data for {mode} loaded.')
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        # Load data and get label
        # X = Image.open(self.df['path'][index])
        X = self.img[index]
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X.float(), y

class HAM10000_ORG(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


def preprocess_df(args):
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # This set will be df_original excluding all rows that are in the val set
    # This function identifies if an image is part of the train or val set.
    def get_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'
    data_dir = args.datadir
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    print(f'Total non duplicate size:{df_undup.shape}')

    # now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=args.seed, stratify=y)
    print(f'From non duplicates, we select validation {df_val.shape}')

    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']
    print(f'Therefore, total training data: {len(df_train)} & validation data : {len(df_val)}')

    print(f'#### Training distribution ####')
    train_dict = df_train['cell_type_idx'].value_counts().to_dict()
    print(sorted(train_dict.items()))

    print(f'#### Testing distribution ####')
    val_dict = df_val['cell_type_idx'].value_counts().to_dict()
    print(sorted(val_dict.items()))
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    return df_train, df_val

def read_preprocessed_df():
    pass

def save_to_pickle(args):
    df_train, df_val = preprocess_df(args)
    # train_list = np.zeros((len(df_train), 32,32,3))
    # for train_idx in tqdm(range(len(df_train))):
    #     train_img = Image.open(df_train['path'][train_idx])
    #     img = train_img.resize((32,32))
    #     img = np.asarray(img)
    #     train_list[train_idx] = img
    # with open('/opt/ml/fl_ham/input/saved_data/train.npy','wb') as f:
    #     np.save(f, train_list)
    # del train_list
    # print(f'training set done')

    val_list = np.zeros((len(df_val), 32 ,32 , 3))
    for val_idx in tqdm(range(len(df_val))):
        val_img = Image.open(df_val['path'][val_idx])
        img = val_img.resize((32, 32))
        img = np.asarray(img)
        val_list[val_idx] = img
    with open('/opt/ml/fl_ham/input/saved_data/eval.npy','wb') as f:
        np.save(f, val_list)
    print(f'validation set done')
    del val_list
if __name__=='__main__':
    import numpy as np
    args = get_args()
    # df_train, df_val = preprocess_df(args)
    #
    # test_img = Image.open(df_train['path'][0])
    # breakpoint()
    # test_npy = np.array(test_img)
    save_to_pickle(args)

