import json
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms



"""
shoot': 426, 'pick': 712, 'block': 996, 'pass': 1070, 'ball in hand': 2362, 'dribble': 3490, 'defense': 3866, 'run': 5924, 'no_action': 6490, 'walk': 11749}
all 37085
0.0114 : 
"""
class BasketballDataset(Dataset):
    """SpaceJam: a Dataset for Basketball Action Recognition.
    
    data_dir: normal and augment
    anno_dir: noraml and augment 
    
    
    Return------------
    sample : dictionary{  video:torch.tensor(action movie) , action: action label one-hot ,class :labels  
    """

    def __init__(self, annotation_dict, augmented_dict, video_dir="./dataset/examples/", 
                 augmented_video_dir="./dataset/augmented-examples/", augment=True, transform=None, poseData=False,joints_to_numpy=False):
        with open(annotation_dict) as f:
            self.anno_list = list(json.load(f).items())
        
        if augment==True:
            self.augment=augment
            with open(augmented_dict)  as f:
                augment_anno_list=list(json.load(f).items())
            self.augmented_video_dir=augmented_video_dir
            self.anno_list.extend(augment_anno_list)
            
        self.video_dir=video_dir
        self.poseData=poseData
        self.transform=transform
        self.augment=augment
        self.joints_to_numpy=joints_to_numpy
        
    def __len__(self):
        return(len(self.anno_list))
    def keystoint(self,x):
        return {int(k): v for k,v in x.items()}
    
    def __getitem__(self,idx):
        video_id=self.anno_list[idx][0]
        #one-hot vector
        encoding=np.squeeze(np.eye(10)[np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1)])
        
        #pose dataがTrueなら姿勢データを読み込む
        if self.poseData and self.augment==False:
            joints=np.load(self.video_dir+video_id+'.npy',allow_pickle=True)
            if self.joints_to_numpy:
                joints=self.joint2numpy(joints=joints)
            sample={'video_id':video_id,'joints':torch.from_numpy(joints) , 'action':torch.from_numpy(
            np.array(encoding[self.anno_list[idx][1]])),'class':self.anno_list[idx][1]}
        else:
            Video=self.VideoToNumpy(video_id)
            sample={'video_id':video_id,'video':torch.from_numpy(Video).float(),
                   'action':torch.from_numpy(np.array(encoding[self.anno_list[idx][1]])),'class':self.anno_list[idx][1]}
        
        return sample
    
    def VideoToNumpy(self,video_id,sampling_rate=1):
        #まず拡張していない動画探索 get video from normal video dir
        video_path=self.video_dir+video_id+'.mp4'
        Video=cv2.VideoCapture(video_path)
        #FRAMES=Video.get(cv2.CAP_PROP_FRAME_COUNT)
        if not Video.isOpened():
            #拡張したdirから探索  from augment video dir
            video_path=self.augmented_video_dir+video_id+'.mp4'
            Video=cv2.VideoCapture(video_path)
        if not Video.isOpened():
            raise Exception('Video file not exist or readable!!')
        
        video_frames=[]
        
        while (Video.isOpened):
            frame_num=1
            ret,frame=Video.read()
            if not ret:
                break
            if  frame_num==1 or (frame_num-1)%sampling_rate==0:
                #print(frame)
                out_frame=np.asarray([frame[..., i] for i in range(frame.shape[-1])]).astype(float)
                #new_frame_3=frame.transpose(2,0,1).astype(float)
                video_frames.append(out_frame)
            frame_num+=1
        Video.release()
        assert len(video_frames)==16
        #return 
        #　動画から16frame 抽出  (c,f,h,w)
        return np.transpose(np.asarray(video_frames),(1,0,2,3))
    
    def joint2numpy(self,joints):
        """
        joints: numpy dictionary    16 frames × 18 joints
        """
        joints_arr=np.zeros([len(joints),18*2],dtype=np.float32)
        for idx in range(len(joints)):
            frame=joints[idx]
            arr=np.array(list(frame.values()))
            #検出してないjointの座標を(0,0)におく
            for i in range(18):
                if i not in frame.keys():
                    arr=np.insert(arr,i,0,axis=0)
            arr=arr.flatten()
            joints_arr[idx]=arr

        return joints_arr