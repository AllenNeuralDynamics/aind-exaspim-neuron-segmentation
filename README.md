# Neuron Segmentation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

A Python package for performing neuron segmentation in ExaSPIM image datasets, designed for large-scale, high-resolution volumetric data. The pipeline combines deep learning–based affinity prediction with graph-based algorithms to produce accurate neuron reconstructions.


## Method

The segmentation pipeline consists of three main steps:

<blockquote>
  <p>1. <strong>Affinity Prediction</strong>: 3D CNN predicts voxel affinities indicating which neighboring voxels belong to the same neuron.</p>
  <p>2. <strong>Watershed Algorithm</strong>: Seeded watershed uses the affinity maps to produce an initial oversegmentation into supervoxels.</p>
  <p>3. <strong>Supervoxel Agglomeration</strong>: Supervoxels are iteratively merged using a graph-based algorithm to form full neuron segments.</p>
</blockquote>

In addition, the repository provides tools for skeletonization and exporting the results as a ZIP archive of SWC files.
<br>
<br>

<p>
  <img src="imgs/pipeline.png" width="850" alt="pipeline">
  <br>
  <b> Figure: </b>Visualization of segmentation pipeline.
</p>

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```
