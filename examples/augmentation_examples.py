import sys
import pathlib
cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))

from skimage.io import imsave

import millipyde as mp

def augment():
    images = mp.images_from_path("examples/images")

    ops = [
        mp.Operation("transpose", probability=.2),
        mp.Operation("fliplr", probability=.2),
        mp.Operation("random_brightness", -1, 1),
        mp.Operation("random_gaussian", 0, 4),
        mp.Operation("rgb2grey", probability=.5),
        mp.Operation("random_rotate", 0, 120, probability = .5)
    ]

    g = mp.Generator(images, ops, return_to_host=True)

    
    for i in range(40):
        img = next(g)
        imsave("examples/augment/dog" + str(i) + ".png", img)
    
def main():
    augment()

if __name__ == '__main__':
    main()
