# Rectified Mask Extractor of Plan2Scene

We use the following 3rd party projects which are imported as git submodules. 
Please agree to their terms of use as well if you use this code.
1) https://github.com/CSAILVision/semantic-segmentation-pytorch
2) https://github.com/svip-lab/PlanarReconstruction 

## How to use?
1) Download the pre-trained checkpoints for 'ade20k-hrnetv2-c1' available [here](http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/). There are two checkpoint files 'encoder_epoch_30.pth' and 'decoder_epoch_30.pth'.
2) Download the pre-trained model for [PlanarReconstruction](https://github.com/svip-lab/PlanarReconstruction/) provided by the authors.
3) Install pre-requisites of [PlanarReconstruction](https://github.com/svip-lab/PlanarReconstruction) and [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch/tree/5c2e9f6f3a231ae9ea150a0019d161fe2896efcf) projects to a common python environement. You can use the provided [environment.yml](environment.yml) file to create a conda environment that has dependencies of both projects. 
    ```bash
    conda env create -f environment.yml
    conda activate plane
    ```

4) Rename `./conf/data-paths-example.json` to `./conf/data-paths.json`. Update checkpoint path fields of it. 
5) Extract rectified surface masks as follows.
    ```bash
    export PYTHONPATH=./code/src
    python ./code/scripts/extract_surface_masks.py ./data/output/floor /path/to/rent3d/dataset/images/ floor
    python ./code/scripts/extract_surface_masks.py ./data/output/wall /path/to/rent3d/dataset/images/ wall
    python ./code/scripts/extract_surface_masks.py ./data/output/ceiling /path/to/rent3d/dataset/images/ ceiling
    ```
    The rectified surface masks are generated at `./data/output/[SURFACE]/rectified_surface_masks` directory.
    You can open `./data/output/[SURFACE]/index.html` to preview the extracted surface masks.cdcdcdcd