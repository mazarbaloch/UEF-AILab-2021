import os, shutil, pathlib

#Path to the directory where the original dataset was uncompressed
original_dir = pathlib.Path("../data/dogs-vs-cats/train")

#Directory where we will store our smaller dataset
new_base_dir = pathlib.Path("../data/cats_vs_dogs_small")

#A function to copy cat and dog images from index start_index to
#index end_index to the subdirectory new_base_dir/{subset_name}/cat
#(respectively dog). "subset_name" will be either "train", "validation", or "test".
def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname, dst=dir / fname)

#Create the training subset with the first 1000 images of each category.
make_subset("train", start_index=0, end_index=1000)

#Create the validation subset with the next 500 images of each category.
make_subset("validation", start_index=1000, end_index=1500)

#Create the test subset with the next 1000 images of each category.
make_subset("test", start_index=1500, end_index=2500)