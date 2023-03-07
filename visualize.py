import numpy as np
from matplotlib import pyplot as plt
import os

envpath = '/home/zzy/anaconda3/envs/torch_gpu_env/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

if __name__ == '__main__':
    feat = np.load("./bot_feats.npy")
    feat = feat.squeeze(0)
    im = np.transpose(feat, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(24):
        ax = plt.subplot(4, 6, i+1)
        cmap = 'jet'
        plt.imshow(im[:, :, i], cmap=plt.get_cmap(cmap))

    plt.show()