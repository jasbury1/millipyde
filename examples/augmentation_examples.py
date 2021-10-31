import sys
import pathlib
cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))

from skimage.io import imsave

import millipyde as mp

def augment():
    images = mp.images_from_path("examples/augment_in")

    ops = [
        mp.Operation("transpose", probability=.2),
        mp.Operation("fliplr", probability=.2),
        mp.Operation("random_brightness", -.2, 1),
        mp.Operation("random_gaussian", 0, .5),
        mp.Operation("random_colorize", [.5, 1.5], [.5, 1.5], [.5, 1.5], probability=.3),
        mp.Operation("rgb2grey", probability=.3),
        mp.Operation("random_rotate", 0, 120, probability = .5)
    ]

    g = mp.Generator(images, ops, return_to_host=True)

    
    for i in range(50):
        img = next(g)
        imsave("examples/augment_out/dog" + str(i) + ".png", img)
    
def main():
    augment()

if __name__ == '__main__':
    main()
