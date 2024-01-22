import numpy as np
from PIL import Image
import torch.utils.data as data

from transformers import RobertaTokenizerFast

from utils_new import SYSU_LABEL2PID, SYSU_Refer, RegDB_Refer, LLCM_Refer, tokenize


TOKEN_PATH = '/data1/dyh/models/RoBERTa/roberta-base-model'


class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.tokenizer = RobertaTokenizerFast.from_pretrained(TOKEN_PATH, local_files_only=True)

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        assert target1 == target2

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        text = SYSU_Refer[SYSU_LABEL2PID[target1]]
        input_ids, attention_mask = tokenize(text, self.tokenizer)

        return dict(
            img1=img1,
            img2=img2,
            text=text,
            target=target1,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def __len__(self):
        return len(self.cIndex)
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label, self.label2pid = load_data(train_color_list)
        thermal_img_file, train_thermal_label, _ = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 384), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 384), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.tokenizer = RobertaTokenizerFast.from_pretrained(TOKEN_PATH, local_files_only=True)

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        assert target1 == target2

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        text = RegDB_Refer[self.label2pid[target1]]
        input_ids, attention_mask = tokenize(text, self.tokenizer)

        return dict(
            img1=img1,
            img2=img2,
            text=text,
            target=target1,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def __len__(self):
        return len(self.cIndex)


class LLCMData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_vis.txt'
        train_thermal_list = data_dir + 'idx/train_nir.txt'

        color_img_file, train_color_label, self.label2pid = load_data(train_color_list)
        thermal_img_file, train_thermal_label, _ = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 384), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 384), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            #print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.tokenizer = RobertaTokenizerFast.from_pretrained(TOKEN_PATH, local_files_only=True)

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        assert target1 == target2

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        text = LLCM_Refer[self.label2pid[target1]]
        input_ids, attention_mask = tokenize(text, self.tokenizer)

        return dict(
            img1=img1,
            img2=img2,
            text=text,
            target=target1,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def __len__(self):
        return len(self.cIndex)
        
        
class TestData(data.Dataset):
    def __init__(self, dataset, test_img_file, test_label, transform=None, img_size = (144,384)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        self.dataset = dataset

        self.tokenizer = RobertaTokenizerFast.from_pretrained(TOKEN_PATH, local_files_only=True)

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)

        if self.dataset == 'sysu':
            text = SYSU_Refer[target1]
        elif self.dataset == 'regdb':
            text = RegDB_Refer[target1]
        elif self.dataset == 'llcm':
            text = LLCM_Refer[target1]

        input_ids, attention_mask = tokenize(text, self.tokenizer)

        return dict(
            img=img1,
            target=target1,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def __len__(self):
        return len(self.test_image)
        

def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

        LABEL2PID = {int(s.split(' ')[1]): int(s.split('/')[1]) for s in data_file_list}
        
    return file_image, file_label, LABEL2PID