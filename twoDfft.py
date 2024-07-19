import os
import numpy as np
import cv2 as cv
import pandas as pd
###2d fft, I dont want to try this because it makes a 2d matrix of frequencies rather than a vector.
# # print(allmovies[0,3,:,:])
# daframe = allmovies[3,3,:,:]
# ffdaframe = np.fft.fft2(daframe)

# ########
# #doesnt work
# # psd = ffdaframe * (np.conj(ffdaframe)/240)
# # plt.plot(psd)
# # plt.show()
# ########
# psd = ffdaframe * np.conj(ffdaframe)/(240**2)
# psd_idx = psd > 10_000
# smallffdaf = ffdaframe*psd_idx
# print('sum is', psd_idx.sum())
# nframe = np.fft.ifft2(smallffdaf)

# print(daframe.shape)
# print('shrink to ', smallffdaf.shape)
# fig, axes = plt.subplots(1,2)
# axes[0].imshow(nframe.real)
# axes[1].imshow(daframe)
# plt.show()