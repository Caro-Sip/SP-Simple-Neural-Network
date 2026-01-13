#!/usr/bin/env python

import os
import struct
from array import array
from os import path
import png

INPUT_DIR = "mnist_ubyte"
OUTPUT_DIR = "mnist_png"


def read(images_file, labels_file):
    with open(labels_file, "rb") as flbl:
        magic, size = struct.unpack(">II", flbl.read(8))
        labels = array("b", flbl.read())

    with open(images_file, "rb") as fimg:
        magic, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = array("B", fimg.read())

    return labels, images, size, rows, cols


def write_dataset(labels, data, size, rows, cols, output_dir):
    for i in range(10):
        os.makedirs(path.join(output_dir, str(i)), exist_ok=True)

    for i, label in enumerate(labels):
        output_filename = path.join(output_dir, str(label), f"{i}.png")
        print("writing", output_filename)

        with open(output_filename, "wb") as f:
            writer = png.Writer(cols, rows, greyscale=True)
            image = [
                data[i * rows * cols + j * cols : i * rows * cols + (j + 1) * cols]
                for j in range(rows)
            ]
            writer.write(f, image)


def main():
    datasets = {
        "train": (
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
        ),
        "test": (
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ),
    }

    for name, (img_file, lbl_file) in datasets.items():
        img_path = path.join(INPUT_DIR, img_file)
        lbl_path = path.join(INPUT_DIR, lbl_file)

        if not path.exists(img_path) or not path.exists(lbl_path):
            print(f"Skipping {name}: files not found")
            continue

        labels, data, size, rows, cols = read(img_path, lbl_path)
        write_dataset(
            labels,
            data,
            size,
            rows,
            cols,
            path.join(OUTPUT_DIR, name),
        )


if __name__ == "__main__":
    main()
