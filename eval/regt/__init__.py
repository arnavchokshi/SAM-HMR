"""GT re-annotation utilities for the count-accuracy eval suite.

The scripts in this subpackage diagnose entry/exit frame drift in
hand-annotated MOT GT files by comparing against high-IDF1 predicted
tracks, then propose a minimal patch that aligns the GT to the actual
video timeline. Phase 0 of the BigTest accuracy work uses this to fix
the documented 2-9 frame entry-time drift on BigTest GT ids 7-14.
"""
