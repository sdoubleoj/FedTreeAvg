import torch
import pickle
import glob
import random
import pdb, os
import torchaudio
import numpy as np
import os.path as osp

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import models, transforms


class feature_manager():
    def __init__(self, args: dict):
        self.args = args
        self.initialize_feature_module()
        
    def initialize_feature_module(self):
        # load gpu or not
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available(): print("GPU available, use GPU")
        
        # load models
        if self.args.feature_type == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier = self.model.classifier[:-1]
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # image transform
            self.img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
    def extract_frame_features(self, 
                               video_id: str, 
                               label_str: str,
                               max_len: int=-1) -> (float, int):
        """Extract the framewise feature from video streams."""
        
        rawframes = os.listdir(Path(self.args.raw_data_dir).joinpath('rawframes', label_str, video_id))
        rawframes.sort()
        if self.args.dataset == "ucf101":
            # downsample to 1 sec per frame
            rawframes = rawframes[::5]
        
        input_data_list = list()
        for rawframe in rawframes:
            rawframe_path = Path.joinpath(Path(self.args.raw_data_dir).joinpath('rawframes', label_str, video_id, rawframe))
            input_image = Image.open(rawframe_path)
            input_tensor = self.img_transform(input_image)
            input_data_list.append(input_tensor.detach().cpu().numpy())
        
        with torch.no_grad():
            input_data = torch.Tensor(np.array(input_data_list)).to(self.device)
            if len(input_data) == 0: return None
            features = self.model(input_data).detach().cpu().numpy()
        if max_len != -1: features = features[:max_len]
        return features
    
    def extract_mfcc_features(self, 
                              audio_path: str, 
                              label_str: str,
                              max_len: int=-1) -> (float, int):
        """Extract the mfcc feature from audio streams."""
        audio, sr = torchaudio.load(str(audio_path))
        features = torchaudio.compliance.kaldi.fbank(
                    waveform=torch.Tensor(torch.Tensor(audio)),
                    frame_length=40, 
                    frame_shift=20,
                    num_mel_bins=80,
                    window_type="hamming")
        features = features.detach().cpu().numpy()
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-5)
        if max_len != -1: features = features[:max_len]
        return features
    
    def fetch_partition(self, fold_idx=1, alpha=0.5):
        # reading partition
        if self.args.dataset == "ucf101":
            alpha_str = str(alpha).replace('.', '')
            partition_path = Path(self.args.output_dir).joinpath("partition", self.args.dataset, f'fold{fold_idx}', f'partition_alpha{alpha_str}.pkl')
        
        with open(str(partition_path), "rb") as f: 
            partition_dict = pickle.load(f)
        return partition_dict