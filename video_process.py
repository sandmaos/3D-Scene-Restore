import os
import cv2
import shutil

import numpy as np
from PIL import ImageEnhance, Image

VIDEO_PATH = 'test.mp4'
EXTRACT_FOLDER = 'video'
EXTRACT_FREQUENCY = 1


def img_enhance(image):

    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)

    enh_col = ImageEnhance.Color(image_brightened)
    color = 1.5
    image_colored = enh_col.enhance(color)



    enh_con = ImageEnhance.Contrast(image_colored)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)



    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)


    return image_sharped

def extract_frames(video_path, dst_folder, index):

    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)
    count = 1
    while True:
        _, frame = video.read()
        frame = Image.fromarray(np.uint8(frame))
        frame = img_enhance(frame)
        if frame is None:
            break
        if count % EXTRACT_FREQUENCY == 0:
            save_path = "{}/{:>03d}.jpg".format(dst_folder, index)
            cv2.imwrite(save_path, np.asarray(frame))
            index += 1
        count += 1
    video.release()

    print("Totally save {:d} pics".format(index - 1))

def main():

    try:
        shutil.rmtree(EXTRACT_FOLDER)
    except OSError:
        pass
    if not os.path.exists(EXTRACT_FOLDER):
        os.mkdir(EXTRACT_FOLDER)

    extract_frames(VIDEO_PATH, EXTRACT_FOLDER, 1)

if __name__ == '__main__':
    main()
