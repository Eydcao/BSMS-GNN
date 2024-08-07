# Updates August 2024
After graduation, the original host (UCLA's Google Drive) for the dataset is unavailable. I took this opportunity to significantly improve the repository as follows:

## Changes:
1. To download and parse the dataset, please refer to the README in the `misc` folder, which contains the full documentation.
2. I now use Poetry for easier environment management. Please see the updated Requirements section.
3. I now use Hydra for handling all configurations, simplifying the process. Please see the updated Usage section.
4. The graph search and BSMS layer generation are now wrapped as clean classes for better readability.
5. The merging of multi-batch graphs is now handled neatly with PyG's Data object.
6. Due to limited time, there is no longer support for contact cases (Fonts and Deformable Plates). Performance may differ from original reports due to the new optimizer, as I did not test it.

## Acknowledgements

I would like to thank [Zijie Huang](https://zijieh.github.io/) for inviting me to help integrate this framework into Nvidia's [Modulus](https://github.com/NVIDIA/modulus) in July 2024. Many of the code revisions in this version were developed during that collaboration project. However, due to time constraints, not all functions, such as multi-batch training, were implemented. This provided the motivation for me to complete the full revision.

# Bi-stride Multi-Scale GNN

This repository contains the code implementations for [Efficient Learning of Mesh-Based Physical Simulation with BSMS-GNN (ICML 2023)](https://openreview.net/forum?id=2Mbo7IEtZW). The paper is also available on [Arxiv](https://arxiv.org/abs/2210.02573).

## Motivations
<div style="display:flex; flex-direction:row;">
    <figure>
        <img src="./figs/motivations.png" height=300px/>
    </figure>
</div>

We focus on developing a multi-scale graph neural network for physics-based simulation. Previous works have certain limitations when it comes to building multi-scale connectivity.

- GraphUNet (Gao et al., 2019) has additional scoring modules to select the most informative nodes for constructing coarser levels. They adopt a power-of-2 adjacency enhancement to prevent loss of connectivity. However, this enhancement does not guarantee connectivity preservation.
- MS-GNN-Grid (Lino et al., 2021) uses background helper grids to build the coarser levels. However, this approach can blur boundaries that are spatially close but not necessarily geodesically close.
- MultiScale MeshGraphNets (Fortunato et al., 2022) uses manually drawn coarser meshes for the same domain, but this requires a significant amount of additional labor.

We aimed to find a solution that would be consistent across any input graphs, without introducing blurring effects on cross-boundary edges, while preserving correct connectivity and minimizing additional labor.

## Method

### Bi-stride pooling

<div style="display:flex; flex-direction:row;">
    <figure>
        <img src="./figs/bi-stride.png" height="192"/>
    </figure>
</div>

We drew inspiration from bipartite graphs, where nodes can be split into two groups, and the minimum geometric distance between the two groups is exactly one hop away. This property allows a simple power-of-2 adjacency enhancement to preserve connectivity. We extend this idea to a general mesh:

1. Select an initial node.
2. Perform Breadth-First-Search on a general mesh, marking the geodesic distance to the initial node.
3. Pool nodes at every other level and apply the power-of-2 adjacency enhancement.

This process ensures that the connectivity is preserved at any depth of coarser level.

### Pipeline

<div style="display:flex; flex-direction:row;">
    <figure>
        <img src="./figs/pipeline.png" height=250px/>
    </figure>
</div>

1. Before training, we employ bi-stride pooling as a pre-processing step to determine the multi-level graph for the input mesh.
2. Based on the multi-scale connectivities, we then determine non-parametric transition modules.
3. These advantages eliminate the need for additional overhead such as scoring modules or matrix enhancement during training.

## Results
Our dataset includes the following: 1) cylinder flow, 2) compressible flow around an airfoil, 3) elastic plate, and 4) inflating elastic surface. The multi-scale structure of these datasets, achieved through bi-stride pooling, is shown below:
<div style="display:flex; flex-direction:row;">
    <div>
    <figure>
        <img src="./figs/examples.png" height="213"/>
    </figure>
    </div>
</div>

The method performs well on all datasets, demonstrating significant improvements in training and inference time as well as RAM consumption.
<div style="display:flex; flex-direction:row;">
    <div>
    <figure>
        <img src="./figs/perform.png" height="150"/>
    </figure>
    </div>
</div>

The absence of cross-boundary edges helps avoid artificial blurring effects.
<div style="display:flex; flex-direction:row;">
    <div>
    <figure>
        <img src="./figs/blur_inter.png" height="125"/>
    </figure>
    </div>
</div>

Bi-stride pooling consistently works on unseen geometry, leading to higher accuracy.
<div style="display:flex; flex-direction:row;">
    <div>
    <figure>
        <img src="./figs/compare_to_learnable.png" height="150"/>
    </figure>
    </div>
</div>

Overall, we achieve the lowest inference error compared to previous methods in the most contact-rich test case.
<div style="display:flex; flex-direction:row;">
    <div>
    <figure>
        <img src="./figs/IDP_error_compare.png" height="150"/>
    </figure>
    </div>
</div>

## Requirements

I now use Poetry for environment management. Setting up is simple:

```sh
conda env create -f environment.yml
conda activate bsms-gnn
poetry install --no-root
```

## Usage

All parameters are now handled by Hydra configuration. Please refer to the configs folder to understand the parameters better.

```sh
# Run a training, defaulting to the airfoil case
# Set board=true to upload info to a WandB board
python src/train.py board=true

# Run a rollout using a specific checkpoint
# Assign the datasets, model to be cylinder_flow
python src/rollout.py restore_dir=./bsms-res/ckpts/train/cylinder_flow/20240806-140921/ restore_step=0 datasets=cylinder_flow model=cylinder_flow
```

## Citation

If you find this method useful, please cite it using the following format:

```latex
@inproceedings{cao2023efficient,
  title     = {Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network},
  author    = {Cao, Yadi and Chai, Menglei and Li, Minchen and Jiang, Chenfanfu},
  booktitle = {International Conference on Machine Learning},
  year      = {2023},
  url       = {https://openreview.net/forum?id=2Mbo7IEtZW}
}
```

## ICML Poster

![poster](./figs/ICML_poster_5k.png)
