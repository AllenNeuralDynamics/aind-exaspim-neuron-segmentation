import time
import numpy as np
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds
import zarr
from volara.blockwise import ExtractFrags
from volara.datasets import Affs, Labels
from volara.dbs import SQLite
import shutil
import os

from volara.blockwise import AffAgglom
from volara.blockwise import GraphMWS
from volara.lut import LUT
from volara.blockwise import Relabel


if __name__ == '__main__':
    # Our input
    raw_aff_path = "YOUR ZARR2 VOLUME HERE.zarr"  # 4D zarr (v2 format)
    aff_path = "/scratch/volara_input.zarr"

    # Volara intermediates
    db_path = "/scratch/test.zarr/db.sqlite"
    frag_path = "/scratch/test.zarr/fragments"
    lut_path = "/scratch/test.zarr/lut"

    # Our output
    seg_path = "/results/segments.zarr"

    # Clean up any existing volara outputs
    paths_to_clean = [
        "/scratch/test.zarr",
        "volara_logs",
    ]
    for path in paths_to_clean:
        if os.path.exists(path):
            shutil.rmtree(path)

    # Volara write input in special format:
    offset = Coordinate(0, 0, 0)
    voxel_size = Coordinate(290, 260, 260)

    affs_array = prepare_ds(
        aff_path,
        (3, 1024, 1024, 1024),
        offset=offset,
        voxel_size=voxel_size,  # 3D voxel size for spatial dimensions
        axis_names=["c^", "z", "y", "x"],  # c^ marks channel as non-spatial
        units=["nm", "nm", "nm"],  # Only spatial dimensions get units
        mode="w",
        dtype=np.float16,
    )

    raw = zarr.open(raw_aff_path, mode='r')
    affs_array[:] = raw[:]

    # Volara abstractions
    db = SQLite(
        path=db_path,
        edge_attrs={
            "zyx_aff": "float",
        },
    )

    affinities = Affs(
        store=aff_path,
        neighborhood=[Coordinate(1, 0, 0), Coordinate(0, 1, 0), Coordinate(0, 0, 1)],
    )

    fragments = Labels(store=frag_path)


    start_time = time.time()
    extract_frags = ExtractFrags(
        db=db,
        affs_data=affinities,
        frags_data=fragments,
        block_size=Coordinate(20, 100, 100),  # Use Coordinate for consistency
        context=Coordinate(2, 2, 2),
        bias=[-0.5, -0.5, -0.5],
    )
    extract_frags.run_blockwise(multiprocessing=True)
    print('Extract Frags Time:', time.time() - start_time)

    # Affinity Agglomeration across blocks
    start_time = time.time()
    aff_agglom = AffAgglom(
        db=db,
        affs_data=affinities,
        frags_data=fragments,
        block_size=(20, 100, 100),
        context=(2, 2, 2),
        scores={"zyx_aff": affinities.neighborhood},
    )
    aff_agglom.run_blockwise(multiprocessing=True)
    print('Aff Agglom Time:', time.time() - start_time)

    # Global MWS
    start_time = time.time()
    global_mws = GraphMWS(
        db=db,
        roi=fragments.array("r").roi,
        lut=LUT(path=lut_path),
        weights={"zyx_aff": (1.0, -0.5)},
        # ^NOTE: Default parameters
    )
    global_mws.run_blockwise(multiprocessing=True)
    print('Global Merge Time:', time.time() - start_time)

    segments = Labels(store=seg_path)

    # Relabel fragments to segments using lut
    start_time = time.time()
    relabel = Relabel(
        frags_data=fragments,
        seg_data=segments,
        lut=LUT(path=lut_path),
        block_size=(20, 100, 100),
    )
    relabel.run_blockwise(multiprocessing=True)
    print('Relabel Time:', time.time() - start_time)