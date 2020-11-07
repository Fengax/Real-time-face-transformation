import cv2
import imageio
import torch
import numpy as np
from animate import normalize_kp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
from demo import make_animation
from skimage import img_as_ubyte
from demo import load_checkpoints
warnings.filterwarnings("ignore")

imageio.plugins.ffmpeg.download()

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

source_image = imageio.imread('02.png')
source_image = resize(source_image,(256,256))[..., :3]


generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                            checkpoint_path='vox-cpk.pth.tar')

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    frame = cv2.flip(frame, 1)
    x = 143
    y = 87
    w = 322
    h = 322
    frame = frame[y:y + h, x:x + w]
    frame = resize(frame, (256, 256))[..., :3]
    source1 = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    kp_driving_initial = kp_detector(source1)
else:
    rval = False

cv2_source = cv2.cvtColor(source_image.astype('float32'),cv2.COLOR_BGR2RGB)
source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
kp_source = kp_detector(source)

predictions = []

while rval:
    driving_frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    driving_frame = driving_frame.cuda()
    kp_driving = kp_detector(driving_frame)
    kp_norm = normalize_kp(kp_source=kp_source,
                           kp_driving=kp_driving,
                           kp_driving_initial=kp_driving_initial,
                           use_relative_movement=True,
                           use_relative_jacobian=True,
                           adapt_movement_scale=True)
    out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
    predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    im = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    joinedFrame = np.concatenate((im, frame), axis=1)
    cv2.imshow("preview", joinedFrame)
    rval, frame = vc.read()
    frame = cv2.flip(frame, 1)
    x = 143
    y = 87
    w = 322
    h = 322
    frame = frame[y:y + h, x:x + w]
    frame = resize(frame, (256, 256))[..., :3]

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
