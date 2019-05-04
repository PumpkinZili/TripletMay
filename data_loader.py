import os
import numpy as np
import pandas as pd
from skimage import io
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import PairwiseDistance

class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets, transform = None):
        
        self.root_dir          = root_dir
        self.df                = pd.read_csv(csv_name)
        self.num_triplets      = num_triplets
        self.transform         = transform
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets)


    @staticmethod
    def generate_triplets(df, num_triplets):
        
        def make_dictionary_for_face_class(df):

            '''
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append(df.iloc[idx, 0])
            return face_classes
        
        triplets    = []
        classes     = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)

        for _ in range(num_triplets):

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''
        
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size = 2, replace = False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append([face_classes[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name])
        
        return triplets
    
    
    def __getitem__(self, idx):
        
        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]
        
        anc_img   = os.path.join(self.root_dir, str(pos_name), str(anc_id) + '.png')
        pos_img   = os.path.join(self.root_dir, str(pos_name), str(pos_id) + '.png')
        neg_img   = os.path.join(self.root_dir, str(neg_name), str(neg_id) + '.png')
        
        # anc_img   = io.imread(anc_img, as_grey=self.is_mnist)
        # pos_img   = io.imread(pos_img, as_grey=self.is_mnist)
        # neg_img   = io.imread(neg_img, as_grey=self.is_mnist)
        anc_img = Image.open(anc_img).convert('RGB')
        pos_img = Image.open(pos_img).convert('RGB')
        neg_img = Image.open(neg_img).convert('RGB')

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))
        
        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class, 'neg_class': neg_class}

        data = [anc_img, pos_img,neg_img]
        label = [pos_class, pos_class, neg_class]

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])
            data = [self.transform(img)
                    for img in data]
            
        # return sample
        return data, label
    
    def __len__(self):
        
        return len(self.training_triplets)


class SemiHardTripletDataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets, features, margin, destination, transform=None):

        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.transform = transform
        self.features = features
        self.margin = margin
        self.destination = destination
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets, self.features, self.margin)

    @staticmethod
    def generate_triplets(df, num_triplets, features, margin):

        def make_dictionary_for_face_class(df):

            '''
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append(df.iloc[idx, 0])
            return face_classes

        triplets = []
        classes = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)
        i = 0
        while i < num_triplets:

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''

            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]
            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            a_position = df.loc[df['class'] == pos_class, 'id'].ix[df['id'] == face_classes[pos_class][ianc]].index.values[0]
            p_position = df.loc[df['class'] == pos_class, 'id'].ix[df['id'] == face_classes[pos_class][ipos]].index.values[0]
            n_position = df.loc[df['class'] == neg_class, 'id'].ix[df['id'] == face_classes[neg_class][ineg]].index.values[0]

            l2_dist = PairwiseDistance(2)
            dp = l2_dist.forward_val(features[a_position], features[p_position])
            dn = l2_dist.forward_val(features[a_position], features[n_position])

            if dp - dn + margin > 0:
                triplets.append(
                    [face_classes[pos_class][ianc], face_classes[pos_class][ipos],
                     face_classes[neg_class][ineg],
                     pos_class, neg_class, pos_name, neg_name])
                i += 1

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir, str(pos_name), str(anc_id) + '.png')
        pos_img = os.path.join(self.root_dir, str(pos_name), str(pos_id) + '.png')
        neg_img = os.path.join(self.root_dir, str(neg_name), str(neg_id) + '.png')
        anc_mea = os.path.join(self.destination, str(pos_name), 'sample1.png')
        neg_mea = os.path.join(self.destination, str(neg_name), 'sample1.png')

        anc_img = Image.open(anc_img).convert('RGB')
        pos_img = Image.open(pos_img).convert('RGB')
        neg_img = Image.open(neg_img).convert('RGB')
        anc_mea = Image.open(anc_mea).convert('RGB')
        neg_mea = Image.open(neg_mea).convert('RGB')

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        # sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'anc_mea':anc_mea, 'neg_mea':neg_mea,
        #           'pos_class': pos_class, 'neg_class': neg_class}

        data = [anc_img, pos_img, neg_img, anc_mea, neg_mea]
        label = [pos_class, pos_class, neg_class, pos_class, neg_class]

        if self.transform:
            # sample['anc_img'] = self.transform(sample['anc_img'])
            # sample['pos_img'] = self.transform(sample['pos_img'])
            # sample['neg_img'] = self.transform(sample['neg_img'])
            # sample['anc_mea'] = self.transform(sample['anc_mea'])
            # sample['neg_mea'] = self.transform(sample['neg_mea'])
            data = [self.transform(img)
                    for img in data]

        return data, label

    def __len__(self):

        return len(self.training_triplets)

