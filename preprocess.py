import os
import json
import pickle
import argparse
import numpy as np

from PIL import Image
from glob import glob
from tqdm import tqdm
from collections import defaultdict

class PreprocessCifar100:
    def __init__(self, data_dir, types):
        self.data_dir = data_dir
        self.types = types
        self.down_err_list = defaultdict(list)
        with open(os.path.join(self.data_dir, 'meta'), 'rb') as f:
            self.meta = pickle.load(f)
        self.label_map = defaultdict(set)
    
    def _unpickle(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data

    def _image_preprocess(self, data_list):
        reshaped_data = np.vstack(data_list).reshape((-1, 3, 32, 32))
        reshaped_data = reshaped_data.swapaxes(1, 3)
        reshaped_data = reshaped_data.swapaxes(1, 2)
        return reshaped_data

    def _label_preprocess(self, coarse_list, fine_list):
        train_labels = [f"{self.meta['coarse_label_names'][coarse_label]}/{self.meta['fine_label_names'][fine_label]}"\
                for fine_label, coarse_label in zip(fine_list, coarse_list)]
        return train_labels

    def _download(self, image_list, label_list, filename_list, type_):
        save_dir = os.path.join(self.data_dir, f"{type_}_images")
        for idx, train_image in tqdm(enumerate(image_list), total=len(image_list)):
            train_image = Image.fromarray(train_image).convert("RGB")
            file_name = filename_list[idx].decode('utf8').replace('.png', '.jpg')
            train_label = label_list[idx]
            coarse_label, fine_label = train_label.split('/')
            self.label_map[coarse_label].add(fine_label)
            
            target_save_dir = os.path.join(save_dir, coarse_label, fine_label)
            if not os.path.exists(target_save_dir):
                os.makedirs(target_save_dir)
            
            save_path = os.path.join(target_save_dir, file_name)
            try:
                train_image.save(save_path, "JPEG")
            except Exception as e:
                self.down_err_list[type_].append({
                    'error_type' : e,
                    'image' : train_image,
                    'file_name' : file_name,
                    'train_label' : train_label,
                    'save_path' : save_path
                })
        
        if type_ in self.down_err_list:
            err_save_dir = os.path.join(self.data_dir, 'down_err')
            if not os.path.exists(err_save_dir):
                os.makedirs(err_save_dir)
            err_save_path = os.path.join(err_save_dir, f"{type_}_error.pickle")
            with open(err_save_path, 'wb') as f:
                pickle.dump(self.down_err_list[type_], f)
    
    def preprocess(self):
        for type_ in self.types:
            print(f"Start preprocess {type_} data")
            data = self._unpickle(os.path.join(self.data_dir, type_))
            image_list = self._image_preprocess(data[b'data'])
            label_list = self._label_preprocess(data[b'coarse_labels'], data[b'fine_labels'])
            filename_list = data[b'filenames']
            self._download(image_list, label_list, filename_list, type_)
            print("Done\n")
        self.label_map = {key:sorted(list(values)) for (key, values) in sorted(self.label_map.items(), key=lambda x : x[0])}
        with open(os.path.join(self.data_dir, 'label_map.json'), 'w') as f:
            json.dump(self.label_map, f, indent=2)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--types', type=str)
    args = parser.parse_args()
    
    args.types = args.types.split(',')

    # cifar100
    if args.dataset == 'cifar100':
        cifar100_data = PreprocessCifar100(
            data_dir='/home/jaeho/hdd/datasets/cifar-100-python',
            types=args.types
        )
        cifar100_data.preprocess()
    else :
        raise 'invalid dataset'