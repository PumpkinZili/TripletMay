import os
import numpy as np
import pandas as pd
import datetime
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import PairwiseDistance


class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, root_dir_preserved, preserved_sample, num_triplets, transform=None):

        self.root_dir = root_dir
        self.root_dir_preserved = root_dir_preserved
        self.df = pd.read_csv(csv_name)
        self.ps = pd.read_csv(preserved_sample)
        self.num_triplets = int(num_triplets/12)
        self.transform = transform
        self.training_triplets = self.generate_triplets(self.df, self.ps, self.num_triplets)

    @staticmethod
    def generate_triplets(df, ps, num_triplets):

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
        classes = ps['class'].unique()
        face_classes = make_dictionary_for_face_class(df)
        preserved_image = make_dictionary_for_face_class(ps)

        for _ in range(num_triplets+1):

            '''
              - all three image is selected from new train set
            '''

            pos_class = np.random.choice([7,8,9])
            neg_class = np.random.choice([7,8,9])
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7,8,9])
            while pos_class == neg_class:
                neg_class = np.random.choice([7,8,9])

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

            triplets.append(
                [face_classes[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name, 1, 1, 1])

        for _ in range(num_triplets * 3):
            '''
                - all three images are selected from preserved images
            '''
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(preserved_image[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = ps.loc[ps['class'] == pos_class, 'name'].values[0]
            neg_name = ps.loc[ps['class'] == neg_class, 'name'].values[0]

            if len(preserved_image[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(preserved_image[pos_class]))
                ipos = np.random.randint(0, len(preserved_image[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(preserved_image[pos_class]))
            ineg = np.random.randint(0, len(preserved_image[neg_class]))

            triplets.append(
                [preserved_image[pos_class][ianc], preserved_image[pos_class][ipos], preserved_image[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name, 2, 2, 2])

        for _ in range(num_triplets * 2):
            '''
                -a and p are from preserved images
                -n is from new train set
            '''
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice([7,8,9])
            while len(preserved_image[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice([7,8,9])

            pos_name = ps.loc[ps['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(preserved_image[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(preserved_image[pos_class]))
                ipos = np.random.randint(0, len(preserved_image[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(preserved_image[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [preserved_image[pos_class][ianc], preserved_image[pos_class][ipos], face_classes[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name, 2, 2, 1])

        for _ in range(num_triplets):

            '''
                -a is from preserved images
                -p and n are from new train set
            '''

            pos_class = np.random.choice([7, 8, 9])
            neg_class = np.random.choice([7, 8, 9])
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7, 8, 9])
            while pos_class == neg_class:
                neg_class = np.random.choice([7, 8, 9])

            pos_name = ps.loc[ps['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(preserved_image[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [preserved_image[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name, 2, 1, 1])

        for _ in range(num_triplets + 1 ):

            '''
                -a and p are from new train set
                -n is from pre 
            '''

            pos_class = np.random.choice([7, 8, 9])
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7, 8, 9])
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = ps.loc[ps['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(preserved_image[neg_class]))

            triplets.append(
                [face_classes[pos_class][ianc], face_classes[pos_class][ipos], preserved_image[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name, 1, 1, 2])

        for _ in range(num_triplets * 2):

            '''
                -a is from new train set
                -p and n is from pre 
            '''

            pos_class = np.random.choice([7, 8, 9])
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7, 8, 9])
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = ps.loc[ps['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(preserved_image[pos_class]))
            ineg = np.random.randint(0, len(preserved_image[neg_class]))

            triplets.append(
                [face_classes[pos_class][ianc], preserved_image[pos_class][ipos], preserved_image[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name, 1, 2, 2])

        for _ in range(num_triplets +1 ):

            '''
                -a and n are from new train set
                -p is from pre 
            '''

            pos_class = np.random.choice([7, 8, 9])
            neg_class = np.random.choice([7, 8, 9])
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7, 8, 9])
            while pos_class == neg_class:
                neg_class = np.random.choice([7, 8, 9])

            pos_name = ps.loc[ps['class'] == pos_class, 'name'].values[0]
            neg_name = ps.loc[ps['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(preserved_image[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [face_classes[pos_class][ianc], preserved_image[pos_class][ipos], face_classes[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name, 1, 2, 1])

        for _ in range(num_triplets + 1):

            '''
                -p is from new train set
                -a and n are from pre 
            '''

            pos_class = np.random.choice([7, 8, 9])
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7, 8, 9])
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = ps.loc[ps['class'] == pos_class, 'name'].values[0]
            neg_name = ps.loc[ps['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(preserved_image[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(preserved_image[neg_class]))

            triplets.append(
                [preserved_image[pos_class][ianc], face_classes[pos_class][ipos], preserved_image[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name, 2, 1, 2])
        return triplets


    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name, a, p, n = self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir if a==1 else self.root_dir_preserved, str(pos_name), str(anc_id) + '.png')
        pos_img = os.path.join(self.root_dir if p==1 else self.root_dir_preserved, str(pos_name), str(pos_id) + '.png')
        neg_img = os.path.join(self.root_dir if n==1 else self.root_dir_preserved, str(neg_name), str(neg_id) + '.png')

        # anc_img   = io.imread(anc_img, as_grey=self.is_mnist)
        # pos_img   = io.imread(pos_img, as_grey=self.is_mnist)
        # neg_img   = io.imread(neg_img, as_grey=self.is_mnist)
        anc_img = Image.open(anc_img).convert('RGB')
        pos_img = Image.open(pos_img).convert('RGB')
        neg_img = Image.open(neg_img).convert('RGB')

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                  'neg_class': neg_class}

        data = [anc_img, pos_img, neg_img]
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

    def __init__(self, root_dir, csv_name, root_dir_preserved, preserved_sample, num_triplets, features, preserved_features, margin, transform=None):

        self.root_dir = root_dir
        self.root_dir_preserved = root_dir_preserved
        self.df = pd.read_csv(csv_name)
        self.ps = pd.read_csv(preserved_sample)
        self.num_triplets = int(num_triplets / 6)
        self.transform = transform
        self.features = features
        self.preserved_features = preserved_features
        self.margin = margin
        self.training_triplets = self.generate_triplets(self.df, self.ps, self.num_triplets, self.features, self.preserved_features, self.margin)

    @staticmethod
    def generate_triplets(df, ps, num_triplets, features, preserved_features, margin):

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
        classes = ps['class'].unique()
        face_classes = make_dictionary_for_face_class(df)
        preserved_image = make_dictionary_for_face_class(ps)
        i = 0
        now_time = datetime.datetime.now()
        while i < num_triplets+1:

            '''
              - all three image is selected from new train set
            '''

            pos_class = np.random.choice([7,8,9])
            neg_class = np.random.choice([7,8,9])
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7,8,9])
            while pos_class == neg_class:
                neg_class = np.random.choice([7,8,9])

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

            triplets.append(
                [face_classes[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name,1, 1, 1])
            i += 1
        i = 0
        last_time = now_time
        now_time = datetime.datetime.now()
        print(now_time - last_time)
        while i < num_triplets + 1:
            '''
                - all three images are selected from preserved images
            '''
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(preserved_image[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = ps.loc[ps['class'] == pos_class, 'name'].values[0]
            neg_name = ps.loc[ps['class'] == neg_class, 'name'].values[0]

            if len(preserved_image[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(preserved_image[pos_class]))
                ipos = np.random.randint(0, len(preserved_image[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(preserved_image[pos_class]))
            ineg = np.random.randint(0, len(preserved_image[neg_class]))

            triplets.append(
                [preserved_image[pos_class][ianc], preserved_image[pos_class][ipos], preserved_image[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name,2,2,2])
            i += 1
        i=0
        last_time = now_time
        now_time = datetime.datetime.now()
        print(now_time - last_time)
        while i < num_triplets + 1:
            '''
                -a and p are from preserved images
                -n is from new train set
            '''
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice([7,8,9])
            while len(preserved_image[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice([7,8,9])

            pos_name = ps.loc[ps['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(preserved_image[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(preserved_image[pos_class]))
                ipos = np.random.randint(0, len(preserved_image[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(preserved_image[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            a_position = pos_class * 3 + ianc
            p_position = pos_class * 3 + ipos
            n_position = \
                df.loc[df['class'] == neg_class, 'id'].ix[df['id'] == face_classes[neg_class][ineg]].index.values[0]

            l2_dist = PairwiseDistance(2)
            dp = l2_dist.forward_val(preserved_features[a_position], preserved_features[p_position])
            dn = l2_dist.forward_val(preserved_features[a_position], features[n_position])

            if dp - dn + margin > 0:
                triplets.append(
                    [preserved_image[pos_class][ianc], preserved_image[pos_class][ipos], face_classes[neg_class][ineg],
                     pos_class, neg_class, pos_name, neg_name,2,2,1])
                i += 1
        i=0
        last_time = now_time
        now_time = datetime.datetime.now()
        print(now_time - last_time)
        while i < num_triplets + 1:
            '''
                -a is from preserved images
                -p and n are from new train set
            '''

            pos_class = np.random.choice([7, 8, 9])
            neg_class = np.random.choice([7, 8, 9])
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7, 8, 9])
            while pos_class == neg_class:
                neg_class = np.random.choice([7, 8, 9])

            pos_name = ps.loc[ps['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(preserved_image[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            a_position = \
                ps.loc[ps['class'] == pos_class, 'id'].ix[ps['id'] == preserved_image[pos_class][ianc]].index.values[0]
            p_position = \
                df.loc[df['class'] == pos_class, 'id'].ix[df['id'] == face_classes[pos_class][ipos]].index.values[0]
            n_position = \
                df.loc[df['class'] == neg_class, 'id'].ix[df['id'] == face_classes[neg_class][ineg]].index.values[0]

            l2_dist = PairwiseDistance(2)
            dp = l2_dist.forward_val(preserved_features[a_position], features[p_position])
            dn = l2_dist.forward_val(preserved_features[a_position], features[n_position])

            if dp - dn + margin > 0:
                triplets.append(
                    [preserved_image[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg],
                     pos_class, neg_class, pos_name, neg_name,2,1,1])
                i += 1
        i=0
        last_time = now_time
        now_time = datetime.datetime.now()
        print(now_time - last_time)
        while i < num_triplets:
            '''
                -a and p are from new train set
                -n is from pre 
            '''

            pos_class = np.random.choice([7, 8, 9])
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7, 8, 9])
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = ps.loc[ps['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(preserved_image[neg_class]))

            a_position = \
                df.loc[df['class'] == pos_class, 'id'].ix[df['id'] == face_classes[pos_class][ianc]].index.values[0]
            p_position = \
                df.loc[df['class'] == pos_class, 'id'].ix[df['id'] == face_classes[pos_class][ipos]].index.values[0]
            n_position = \
                ps.loc[ps['class'] == neg_class, 'id'].ix[ps['id'] == preserved_image[neg_class][ineg]].index.values[0]

            l2_dist = PairwiseDistance(2)
            dp = l2_dist.forward_val(features[a_position], features[p_position])
            dn = l2_dist.forward_val(features[a_position], preserved_features[n_position])

            if dp - dn + margin > 0:
                triplets.append(
                    [face_classes[pos_class][ianc], face_classes[pos_class][ipos], preserved_image[neg_class][ineg],
                     pos_class, neg_class, pos_name, neg_name,1,1,2])
                i += 1
        i=0
        last_time = now_time
        now_time = datetime.datetime.now()
        print(now_time - last_time)
        while i < num_triplets:
            '''
                -a is from new train set
                -p and n is from pre 
            '''

            pos_class = np.random.choice([7, 8, 9])
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice([7, 8, 9])
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = ps.loc[ps['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(preserved_image[pos_class]))
            ineg = np.random.randint(0, len(preserved_image[neg_class]))

            a_position = \
                df.loc[df['class'] == pos_class, 'id'].ix[df['id'] == face_classes[pos_class][ianc]].index.values[0]
            p_position = \
                ps.loc[ps['class'] == pos_class, 'id'].ix[ps['id'] == preserved_image[pos_class][ipos]].index.values[0]
            n_position = \
                ps.loc[ps['class'] == neg_class, 'id'].ix[ps['id'] == preserved_image[neg_class][ineg]].index.values[0]

            l2_dist = PairwiseDistance(2)
            dp = l2_dist.forward_val(features[a_position], preserved_features[p_position])
            dn = l2_dist.forward_val(features[a_position], preserved_features[n_position])

            if dp - dn + margin > 0:
                triplets.append(
                    [face_classes[pos_class][ianc], preserved_image[pos_class][ipos], preserved_image[neg_class][ineg],
                     pos_class, neg_class, pos_name, neg_name,1,2,2])
                i += 1
        last_time = now_time
        now_time = datetime.datetime.now()
        print(now_time - last_time)
        return triplets


    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name,a,p,n = self.training_triplets[idx]
        anc_img = os.path.join(self.root_dir if a==1 else self.root_dir_preserved, str(pos_name), str(anc_id) + '.png')
        pos_img = os.path.join(self.root_dir if p==1 else self.root_dir_preserved, str(pos_name), str(pos_id) + '.png')
        neg_img = os.path.join(self.root_dir if n==1 else self.root_dir_preserved, str(neg_name), str(neg_id) + '.png')

        # anc_img = io.imread(anc_img, as_grey=self.is_mnist)
        # pos_img = io.imread(pos_img, as_grey=self.is_mnist)
        # neg_img = io.imread(neg_img, as_grey=self.is_mnist)
        anc_img = Image.open(anc_img).convert('RGB')
        pos_img = Image.open(pos_img).convert('RGB')
        neg_img = Image.open(neg_img).convert('RGB')

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                  'neg_class': neg_class}

        data = [anc_img, pos_img, neg_img]
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

