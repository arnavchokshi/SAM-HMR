"""Stage D — comparison harness for the dual 3D pipeline.

Reads ``joints_world.npy`` from both pipelines, harmonises them to a
common COCO-17 layout, computes jitter / MPJPE / foot-skating metrics,
and renders a side-by-side overlay video.
"""
