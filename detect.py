import argparse
import os
import sys

import imageio
import numpy as np
import tensorflow as tf

sys.path.append("research")
sys.path.append("research/slim")

from object_detection.utils import label_map_util

MAX_NUM_CLASSES = 100
FACE_CLASS_ID = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', type=str, help='Path to the *.pb file.',
        default='models/ssd/original_single_8bits_2019-02-02_01_33_02_graph.pb')
    parser.add_argument('-l', '--labels', type=str, help='Path to the label_map.pbtxt file.',
        default='models/ssd/label_map.pbtxt')
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='Path to directory with  images/ and annotations/.')
    parser.add_argument('-o', '--output', type=str, help='Path to output dir.',
        default='output')
    return parser.parse_args()


def load_graph(graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def get_images(images_dir):
    images = []
    for image in os.listdir(images_dir):
        images.append(os.path.join(images_dir, image))
    return images


def convert_img_to_rgb(img):
    """
    Converts image img to 3 channels (intended for grey scale to RGB conversion)
    :param img:
    :return:
    """
    if len(img.shape) != 3:
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = img
        ret[:, :, 1] = img
        ret[:, :, 2] = img
        img = ret
    return img


def extract_face_info(boxes, classes, scores):
    boxes = boxes[classes==FACE_CLASS_ID]
    scores = scores[classes == FACE_CLASS_ID]
    classes = classes[classes == FACE_CLASS_ID]

    boxes = boxes[scores > 0.5]
    classes = classes[scores > 0.5]
    scores = scores[scores > 0.5]
    return boxes, classes, scores


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    detection_graph = load_graph(args.graph)
    label_map = label_map_util.load_labelmap(args.labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=MAX_NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    object_scores = {'eye': [], 'nose': [], 'face': []}
    imgs = get_images(args.input_dir)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for img in imgs:
                image = convert_img_to_rgb(imageio.imread(img))
                yimg = image.shape[0]
                ximg = image.shape[1]

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes).astype(np.int32)
                scores = np.squeeze(scores)
                num_detections = np.squeeze(num_detections.astype(int))

                boxes, classes, scores = extract_face_info(boxes, classes, scores)

                if len(boxes) > 1:
                    print('Detected more than one face ({})'.format(img))
                for box in boxes:
                    crop = image[int(box[0] * yimg):int(box[2] * yimg), int(box[1] * ximg):int(box[3] * ximg)]
                    crop_path = os.path.join(args.output, img.split('/')[-1])
                    imageio.imsave(crop_path, crop)
                    print('Face visible on {} was saved in {}'.format(img, crop_path))

