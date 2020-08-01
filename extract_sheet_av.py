import numpy as np
import os
from PIL import Image
from tqdm import trange, tqdm
import av


def l2_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    y = (x1 - x2) ** 2
    return y.sum()


def save_frame(x: np.ndarray, out_dir: str, index: int):
    img = Image.fromarray(x)
    img.save(f'{out_dir}/{index}.webp', lossless=True)


def main():
    video_path = 'Misty. Arranged for solo piano, with music sheet.-vhRwSAbA-CQ.webm'
    threshold = 10000000.0
    out_dir = 'out-webp'

    os.makedirs(out_dir, exist_ok=True)

    container = av.open(video_path)

    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'

    counter = 0
    prev_npframe = None
    for frame in tqdm(container.decode(stream)):
        npframe = frame.to_ndarray(format='rgb24')

        if prev_npframe is None:
            prev_npframe = npframe
            save_frame(npframe, out_dir, counter)

        if l2_distance(npframe, prev_npframe) > threshold:
            counter += 1
            save_frame(npframe, out_dir, counter)
            prev_npframe = npframe


if __name__ == "__main__":
    main()
