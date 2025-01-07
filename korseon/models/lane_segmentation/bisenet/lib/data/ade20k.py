#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import bisenet.lib.data.transform_cv2 as T
from bisenet.lib.data.base_dataset import BaseDataset

'''
proportion of each class label pixels:
    [0.1692778570779725, 0.11564757275917185, 0.0952101638485813, 0.06663867349694136, 0.05213595836428788, 0.04856869977177328, 0.04285300460652723, 0.024667459730413076, 0.021459432596108052, 0.01951911788079975, 0.019458422169334556, 0.017972951662770457, 0.017102797922112795, 0.016127154995430226, 0.012743318904507446, 0.011871312183986243, 0.01169223174996906, 0.010873715499098895, 0.01119535711707017, 0.01106824347921356, 0.010700814956159628, 0.00792769980935508, 0.007320940186670243, 0.007101978087028939, 0.006652130884336369, 0.0065129268341813954, 0.005905601374046595, 0.005655465856321791, 0.00485152244584825, 0.004812313401121428, 0.004808430157907591, 0.004852065319115992, 0.0035166264746248105, 0.0034049293812196796, 0.0031501695661207163, 0.003200865983720736, 0.0027563053654176255, 0.0026019635559833536, 0.002535207367187799, 0.0024709898687369503, 0.002511264681160722, 0.002349575022340693, 0.0022952289072600395, 0.0021756144527500325, 0.0020667410351909894,
    0.002019785482875027, 0.001971430263652598, 0.0019830032929254865, 0.0019170129596070547, 0.0019400873699042965, 0.0019177214046286212, 0.001992758707175458, 0.0019064211898405371, 0.001794991169874655, 0.0017086228805355563, 0.001816450049952539, 0.0018115561530790863, 0.0017526224833158293, 0.0016693853602227783, 0.001690968246884664, 0.001672815290479542, 0.0016435338913693607, 0.0015994805524026869, 0.001415586825791652, 0.0015309535955159497, 0.0015066783881302896, 0.0015584265652761034, 0.0014294452504793305, 0.0014381224963739522, 0.0013854752714941247, 0.001299217899155161, 0.0012526667460881378, 0.0013178209535318454, 0.0012941402888239277, 0.0010893388225083507, 0.0011300189527483507, 0.0010488809855522653, 0.0009206912461167046, 0.0009957668988478528, 0.0009413381127111981, 0.0009365154048026355, 0.0009059601825045681, 0.0008541199189880419, 0.0008971791385063005, 0.0008428502465623139, 0.0008056902958152122, 0.0008098830962054097, 0.0007822564960661871, 0.0007982742428082544, 0.0007502832355158758, 0.0007779780392762995, 0.0007712568824233966, 0.0007453305503359334, 0.0006837047894907241, 0.0007144561259049724, 0.0006892632697976981,
    0.0006652429648347085, 0.0006708271650257716, 0.0006737982709217282, 0.0006266153732017621, 0.0006591083131957701, 0.0006729084088606035, 0.0006615025588342957, 0.0005978453864296776, 0.0005662905332794616, 0.0005832571600309656, 0.000558171776296493, 0.0005270943484946844, 0.0005918616094679417, 0.0005653340750898915, 0.0005626451989934503, 0.0005906185582842337, 0.0005217418569022469, 0.0005282586325333688, 0.0005198277923139954, 0.0004861910064034809, 0.0005218504774841597, 0.0005172358250665335, 0.0005247616468645153, 0.0005357304885031275, 0.0004276964118043196, 0.0004607179872730913, 0.00041193838996318965, 0.00042133234798497776, 0.000374820234027733, 0.00041071531761801536, 0.0003664373889492048, 0.00043033958917813777, 0.00037797413481418125, 0.0004129435322190717, 0.00037504252731164754, 0.0003633328611545351, 0.00039741354470741193, 0.0003815260048785467, 0.00037395769934345317, 0.00037914990094397704, 0.000360210650939554, 0.0003641708241638368, 0.0003354311501122861, 0.0003386525655944687, 0.0003593692433029189, 0.00034422115014162057, 0.00032131529694189243, 0.00031263024322531515, 0.0003252564098949305, 0.00034751306566322646, 0.0002711341955909471, 0.00022987904222809388, 0.000242549759411221, 0.0002045743505533957]
'''



class ADE20k(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(ADE20k, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 150
        self.lb_ignore = 255
        self.lb_map = np.arange(200) - 1 # label range from 1 to 149, 0 is ignored
        self.lb_map[0] = 255

        self.to_tensor = T.ToTensor(
            mean=(0.49343230, 0.46819794,  0.43106043), # ade20k, rgb
            std=(0.25680755, 0.25506608, 0.27422913),
        )

