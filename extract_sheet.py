import decord
import numpy as np
import os
from PIL import Image
from tqdm import trange


decord.bridge.set_bridge('torch')

def l2_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    y = (x1 - x2) ** 2
    return y.sum()


def save_frame(x: np.ndarray, out_dir: str, index: int):
    img = Image.fromarray(x)
    img.save(f'{out_dir}/{index}.png')


def main():
    video_path = 'Misty. Arranged for solo piano, with music sheet.-vhRwSAbA-CQ.webm'
    threshold = 100000.0
    out_dir = 'out'

    os.makedirs(out_dir, exist_ok=True)

    vr = decord.VideoReader(video_path)
    fps = int(vr.get_avg_fps())
    num_frames = len(vr)
    print('Total frames:', num_frames, 'fps:', fps)

    # threshold = 1.0

    # for x in vr[0].shape[:2]:
    #     threshold *= x

    print('Threshold:', threshold)

    start_frame = 10
    prev_frame = vr.get_batch([start_frame])[0].numpy()

    counter = 0
    save_frame(prev_frame, out_dir, counter)
    for index in trange(start_frame + fps, num_frames, fps):
        frame = vr.get_batch([index])[0].numpy()
        # if l2_distance(frame, prev_frame) > threshold:
        #     print('Index:', index)
        #     counter += 1
        #     save_frame(frame, out_dir, counter)
        #     prev_frame = frame


if __name__ == "__main__":
    main()
