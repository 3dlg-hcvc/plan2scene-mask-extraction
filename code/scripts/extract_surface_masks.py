import argparse
import sys
import os
import os.path as osp
import numpy as np
from PIL import Image
from config_parser import parse_config
import logging
from surface_rectification import get_largest_inscribed_square, get_rectified_mask
from module_loader import load_plane_predictor, load_surface_segmenter

HTML_HEAD = "<head>\n <style>\n table, td, th {\n   border: 1px solid black;\n }\n \n table {\n   border-collapse: collapse;\n   width: 100%;\n }\n \n th {\n   height: 50px;\n }\n </style>\n </head>\n"

if __name__ == "__main__":
    # Inputs
    parser = argparse.ArgumentParser(
        description="Extract surface masks from photos.")
    parser.add_argument("output_path",
                        help="Target directory to store extracted surface masks.")
    parser.add_argument("photos_path",
                        help="Path to directory containing photos.")
    parser.add_argument("surface_type",
                        help="Surface type which could be floor, walls or ceiling.")
    parser.add_argument("-l", "--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default="INFO",
                        help="Set the log level")
    
    parser.add_argument("--data-paths", help="Path to ./conf/data_paths.json", default="./conf/data-paths.json")
    parser.add_argument("--masks-conf-path", help="Path to ./conf/plan2scene-masks.json", default="./conf/plan2scene-masks.json")

    parser.add_argument("--largest-surface-count", default=-1, help="Number of largest surface instances to be extracted. \
                                                                  Default arguement depends on surface type (10 for walls and \
                                                                  0 for other surfaces). \
                                                                  Pass 0 to skip instance segmentation.")


    args = parser.parse_args()
    logging.basicConfig(level=logging.getLevelName(args.log_level))
    
    # Configs
    output_path = osp.abspath(args.output_path)
    interested_class_name = args.surface_type
    photos_path = osp.abspath(args.photos_path)
    data_paths_conf = parse_config(args.data_paths)
    masks_conf = parse_config(args.masks_conf_path)
    semantic_segmentation_project_path = osp.abspath(data_paths_conf.semantic_segmentation_project_path)
    planar_reconstruction_project_path = osp.abspath(data_paths_conf.planar_reconstruction_project_path)
    surface_encoder_checkpoint = osp.abspath(data_paths_conf.semantic_segmentation_encoder_checkpoint)
    surface_decoder_checkpoint = osp.abspath(data_paths_conf.semantic_segmentation_decoder_checkpoint)
    planar_checkpoint_dir = osp.abspath(data_paths_conf.planar_reconstruction_checkpoint)

    # Set the max count of plane instances that should be detected
    largest_surface_count = args.largest_surface_count
    if largest_surface_count == -1:
        largest_surface_count = masks_conf.plane_instance_segmentation_count[args.surface_type]

    # Dataset: Code adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch
    dataset_cfg = masks_conf.dataset_cfg
    class_index_map = masks_conf.class_index_map # We use this to identify interested surfaces

    horizontal_fov = masks_conf.fov.horizontal / 180.0 * np.pi
    vertical_fov = masks_conf.fov.vertical / 180.0 * np.pi

    # Output paths
    masks_path = osp.join(output_path, "surface_masks")
    rectified_masks_path = osp.join(output_path, "rectified_surface_masks")

    instance_segment = False
    if largest_surface_count > 0: # Segmentation of planes into instances is enabled.
        instance_segment = True
    interested_class_index = class_index_map[interested_class_name]


    # Import planner reconstruction project
    PlanePredictor = load_plane_predictor(planar_reconstruction_project_path)
    masks_conf.plane_prediction_cfg.resume_dir = planar_checkpoint_dir
    plane_predictor = PlanePredictor(masks_conf.plane_prediction_cfg)
    
    # Import Semantic Segmentation PyTorch project
    SurfaceSegmenter = load_surface_segmenter(semantic_segmentation_project_path)
    surface_segmenter = SurfaceSegmenter(python_semantic_segmentation_path=semantic_segmentation_project_path,
                                         surface_encoder_checkpoint=surface_encoder_checkpoint,
                                         surface_decoder_checkpoint=surface_decoder_checkpoint,
                                         dataset_cfg=dataset_cfg)


    from data_readers.image_data_reader import ImageDataset
    dataset_test = ImageDataset(photos_path, dataset_cfg)

    if not os.path.exists(masks_path):
        os.makedirs(masks_path)

    if not os.path.exists(rectified_masks_path):
        os.makedirs(rectified_masks_path)

    # Show results in an HTML page
    with open(osp.join(output_path, "index.htm"), "w") as f:
        f.write(HTML_HEAD)
        f.write("<table>\n")
        f.write("<tr><th>Image</th><th>Index</th><th>Mask</th><th>Rectified Mask</th></tr>")
        for i, selected_img in enumerate(dataset_test):
            output_name = selected_img["info"].replace(photos_path + "/", "").replace("/", "_").replace(".jpg", "")
            if os.path.exists(os.path.join(masks_path, output_name + ".done")):
                logging.info("Skipping %d/%d %s"% (i, len(dataset_test), output_name))
                continue
            logging.info("Processing %d/%d: %s" % (i, len(dataset_test), output_name))

            img, pred = surface_segmenter.predict_surfaces(selected_img)
            interested_mask = pred == interested_class_index
            
            depth_np, colours_np, instance_parameter, predict_segmentation_np, param_np, _ = plane_predictor.predict_planes(
                    selected_img["info"])
            
            if instance_segment:
                # Identify largest surfaces
                area_list = []
                for current_plane_index in range(15): # Planer Reconstructions detects upto 15 masks
                    common_mask = interested_mask * (predict_segmentation_np == current_plane_index)
                    area = np.sum(common_mask)
                    if area > 0:
                        area_list.append((current_plane_index, common_mask, area))

                area_list = sorted(area_list, key=lambda a: a[2], reverse=True)
                area_list = area_list[:largest_surface_count]
            else:
                area_list = [(0, interested_mask, np.sum(interested_mask))]

            new_plane_index = 0
            for current_plane_index, common_mask, area in area_list:
                logging.info("[%d/%d] %s : Plane %d" % (i, len(dataset_test), output_name, current_plane_index))
                
                if common_mask.max() == False:
                    continue

                img_interested = Image.fromarray(selected_img['img_ori'].copy())
                img_interested.putalpha(Image.fromarray(common_mask))
                img_interested.save(os.path.join(masks_path, output_name + "_" + str(new_plane_index) + ".png"))
                img.save(os.path.join(masks_path, output_name + "_" + str(new_plane_index) + "_image.png"))

                inscribed_rect = get_largest_inscribed_square(common_mask)

                if inscribed_rect[0] > 0:
                    rectified, _ = get_rectified_mask(img_interested, colours_np, depth_np, instance_parameter, param_np, predict_segmentation_np, inscribed_rect,vertical_fov = vertical_fov, horizontal_fov=horizontal_fov)
                    rectified.save(os.path.join(rectified_masks_path, output_name + "_" + str(new_plane_index) + ".png"))

                # Print HTML Table Row
                f.write("<tr><td><img src='surface_masks/" + output_name + "_" + str(
                    new_plane_index) + "_image.png" + "'></td>")
                f.write("<td>" + str(new_plane_index) + "</td>")
                f.write("<td><img src='surface_masks/" + output_name + "_" + str(
                    new_plane_index) + ".png" + "'></td>")
                f.write("<td><img src='rectified_surface_masks/" + output_name + "_" + str(
                    new_plane_index) + ".png" + "'></td>")
                f.write("</tr>\n")
                f.flush()
                new_plane_index += 1

            with open(os.path.join(masks_path, output_name + ".done"), "w") as g:
                g.write("done")