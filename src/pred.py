import os

import cv2
import numpy as np
import torch

from unet_model import UNet
from utils import predict_img

if __name__ == "__main__":
    net = UNet(n_channels=3, n_classes=2, bilinear=False)

    cap = cv2.VideoCapture("video/video.mp4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打开视频文件并读取第一帧
    ret, frame = cap.read()

    # 获取视频帧率和尺寸
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频编写器并写入第一帧
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("tank.mp4", fourcc, fps, (width, height))
    writer.write(frame)

    net.to(device=device)
    state_dict = torch.load("model/checkpoint_epoch5.pth", map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    count = 0
    framecount = 0

    targetimg_l = []
    targetimg_foldpath = "video/dessert"
    for filename in os.listdir(targetimg_foldpath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(targetimg_foldpath, filename)
            targetimg = cv2.resize(cv2.imread(file_path), (width, height), interpolation=cv2.INTER_CUBIC)
            targetimg_l.append(targetimg)

    while True:
        ok, frame = cap.read()
        if ok:
            print(f"{framecount:d}")
            framecount += 1

            img = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR,
            )
            mask = predict_img(
                net=net,
                full_img=img,
                width=width,
                height=height,
                scale_factor=0.75,
                out_threshold=0.5,
                device=device,
            )

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Replace pixels in original image with corresponding pixels in new image
            img[mask != 0] = targetimg_l[0][mask != 0]
            cv2.imwrite(f"result/tank/{framecount}.jpg", img)
            writer.write(img)
            count = (count + 1) % 300
        else:
            break
    writer.release()
    cap.release()
