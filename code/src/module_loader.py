import sys
def load_plane_predictor(planar_reconstruction_path):
    sys.path.append(planar_reconstruction_path)

    from surface_mask_extraction.plane_prediction import PlanePredictor
    module = PlanePredictor
    sys.path.remove(planar_reconstruction_path)

    # Unload modules that conflict with semantic-segmentation-pytorch
    del sys.modules["models"] 
    del sys.modules["utils"]
    return module


def load_surface_segmenter(semantic_segmentation_project_path):
    sys.path.append(semantic_segmentation_project_path)
    from surface_mask_extraction.surface_segmentation import SurfaceSegmenter
    module = SurfaceSegmenter

    # Unload modules that conflict with PlanarReconstruction
    del sys.modules["models"]
    del sys.modules["utils"]

    return module