import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
from demo import make_animation
from skimage import img_as_ubyte
from demo import load_checkpoints
warnings.filterwarnings("ignore")

imageio.plugins.ffmpeg.download()

source_image = imageio.imread('02.png')
reader = imageio.get_reader('04.mp4')


print("resizing image and video")

source_image = resize(source_image, (256, 256))[..., :3]

fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

print("loading checkpoints")


generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', 
                            checkpoint_path='vox-cpk.pth.tar')

print("animating image")



predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

print("saving video, changed")

#save resulting video
imageio.mimsave('generated.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)
#video can be downloaded from /content folder
