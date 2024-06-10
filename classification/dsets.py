import copy
import csv
import functools
import glob
import os
import random
import math
import SimpleITK as sitk
from collections import namedtuple

import torch
import torch.nn.functional as F
from cancerDetection.util.util import XyzTuple, xyz2irc

import numpy as np
import diskcache

CandidateInfoTuple = namedtuple('CandidateInfoTuple','isNodule,diameter_mm,series_uid,center_xyz')
base_path = 'E:\luna16\\'
annotation_file_path = 'CSVFILES\\annotations.csv'
candidate_file_path = 'CSVFILES\\candidates.csv'
cache_path = "E:\luna16\cache\\"
disk_cache = diskcache.FanoutCache(cache_path)


@functools.lru_cache(1)
def getCandidateInforList(requireOnDisk_bool = True):
    mhd_list = glob.glob(os.path.join(base_path,'subset*\*.mhd'))
    setPresentOnDistk = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open(os.path.join(base_path,annotation_file_path),"r") as f:
        for line in list(csv.reader(f))[1:]:
            series_uid = line[0]
            annotationCenter_xyz = tuple([float(c) for c in line[1:4]])
            annotationCenter_diameter = float(line[4])
            diameter_dict.setdefault(series_uid,[]).append(
                (annotationCenter_xyz,annotationCenter_diameter)
            )

    candidateInforList = []
    with open(os.path.join(base_path,candidate_file_path),'r') as f:
        for line in list(csv.reader(f))[1:]:
            series_uid = line[0]
            if series_uid not in setPresentOnDistk and requireOnDisk_bool:
                continue
            candidate_xyz = tuple([float(c) for c in line[1:4]])
            isnodule = bool(int(line[4]))

            candidate_diameter = 0.0
            for annotationCenter_tuple in diameter_dict.get(series_uid,[]):
                annotationCenter_xyz, annotationCenter_diameter = annotationCenter_tuple
                issamenodule = True
                for i in range(3):
                    delta_mm = abs(annotationCenter_xyz[i]-candidate_xyz[i])
                    if delta_mm > annotationCenter_diameter/4:
                        issamenodule = False
                if issamenodule:
                    candidate_diameter = annotationCenter_diameter
            candidateInforList.append(CandidateInfoTuple(
                isnodule,
                candidate_diameter,
                series_uid,
                candidate_xyz
            ))
    candidateInforList.sort(reverse=True)
    return candidateInforList

class CT:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'E:\luna16\subset*/{}.mhd'.format(series_uid)
        )[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc

@functools.lru_cache(1,typed=True)
def getCt(series_uid):
    return CT(series_uid)

@disk_cache.memoize(typed=True)
def getCtRawCandidate(series_uid,center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk,center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk,center_irc

def getCtAugmentedCandidate(
        augmentation_dict,
        series_uid, center_xyz, width_irc):
    ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)
    # ... <1>

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float


    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            ct_t.size(),
            align_corners=False,
        )

    augmented_chunk = F.grid_sample(
            ct_t,
            affine_t,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc

class LunaDataset:
    def __init__(self,
                 isValSet_bool = None,
                 val_stride = 0,
                 ratio_int = 0,
                 series_uid = None,
                 augmentation_dict = None):
        self.candidateinfo_list = copy.copy(getCandidateInforList())
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict
        if series_uid:
            self.candidateinfo_list = [x for x in self.candidateinfo_list if x.series_uid == series_uid]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateinfo_list = self.candidateinfo_list[::val_stride]
            assert self.candidateinfo_list
        elif val_stride > 0:
            del self.candidateinfo_list[::val_stride]
            assert self.candidateinfo_list

        if self.ratio_int:
            self.pos_list = [pos for pos in self.candidateinfo_list if pos.isNodule]
            self.neg_list = [neg for neg in self.candidateinfo_list if not neg.isNodule]

    def __len__(self):
        return len(self.candidateinfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int +1)

            if ndx % (self.ratio_int +1):
                neg_ndx = ndx - pos_ndx - 1
                neg_ndx = neg_ndx % len(self.neg_list)
                candidateinfo_tuple = self.candidateinfo_list[neg_ndx]
            else:
                pos_ndx = pos_ndx % len(self.pos_list)
                candidateinfo_tuple = self.candidateinfo_list[pos_ndx]
        else:
            candidateinfo_tuple = self.candidateinfo_list[ndx]
        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateinfo_tuple.series_uid,
                candidateinfo_tuple.center_xyz,
                width_irc
            )
        else:
            candidate_a, center_irc = getCtRawCandidate(candidateinfo_tuple.series_uid,candidateinfo_tuple.center_xyz,width_irc)
            candidate_t = torch.from_numpy(candidate_a)
            candidate_t = candidate_t.unsqueeze(0)
        candidate_t = candidate_t.to(torch.float32)


        pos_t = torch.tensor([
            not candidateinfo_tuple.isNodule,
            candidateinfo_tuple.isNodule
        ],dtype=torch.long)

        return(candidate_t,
               pos_t,
               candidateinfo_tuple.series_uid,
               torch.tensor(center_irc))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.pos_list)
            random.shuffle(self.neg_list)

