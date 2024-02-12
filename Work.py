#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import streamlit as st

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME)

}
files = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5200)])
    except RunTimeError as e:
        st.text(e)
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


import cv2
import numpy as np
from matplotlib import pyplot as plt
st.title("License Plate Recognition System")
# get_ipython().run_line_magic('matplotlib', 'inline')
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
uploaded_file = st.file_uploader("Upload Image", type=[ 'jpeg', 'jpg', 'png'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Process the image
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the image on the figure
    ax.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))

    # Show the figure using st.pyplot
    st.pyplot(fig)

    # plt.show()
    import easyocr

    detection_threshold = 0.7
    image = image_np_with_detections
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    width = image.shape[1]
    height = image.shape[0]
    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        st.text(box)
        roi = box * [height, width, height, width]
        st.text(roi)
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)
        st.text(ocr_result)
        st.image(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))

        # st.pyplot(plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)))
    # for result in ocr_result:
    if 'ocr_result' in locals() and ocr_result:
        for result in ocr_result:
            difference_sum = np.sum(np.subtract(result[0][2], result[0][1]))
            text = result[1]

            # Display the difference sum and OCR text
            st.text("Difference Sum: " + str(difference_sum))
            st.text("OCR Text: " + text)

            st.text("---")  # Add a separator between each result
        region_threshold = 0.05


        def filter_text(region, ocr_result, region_threshold):
            rectangle_size = region.shape[0] * region.shape[1]

            plate = []
            for result in ocr_result:
                length = np.sum(np.subtract(result[0][1], result[0][0]))
                height = np.sum(np.subtract(result[0][2], result[0][1]))

                if length * height / rectangle_size > region_threshold:
                    plate.append(result[1])
            return plate


        filter_text(region, ocr_result, region_threshold)
        st.text("OCR Text:" + text)
        region_threshold = 0.6


        def ocr_it(image, detections, detection_threshold, region_threshold):

            # Scores, boxes and classes above threhold
            scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
            boxes = detections['detection_boxes'][:len(scores)]
            classes = detections['detection_classes'][:len(scores)]

            # Full image dimensions
            width = image.shape[1]
            height = image.shape[0]

            # Apply ROI filtering and OCR
            for idx, box in enumerate(boxes):
                roi = box * [height, width, height, width]
                region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
                reader = easyocr.Reader(['en'])
                ocr_result = reader.readtext(region)

                text = filter_text(region, ocr_result, region_threshold)
                st.image(region, channels="RGB")

                # st.pyplot(plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)))
                # plt.show()
                st.text(text)
                return text, region


        text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
        import csv
        import uuid

        '{}.jpg'.format(uuid.uuid1())


        def save_results(text, region, csv_filename, folder_path):
            img_name = '{}.jpg'.format(uuid.uuid1())

            cv2.imwrite(os.path.join(folder_path, img_name), region)

            with open(csv_filename, mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([img_name, text])


        save_results(text, region, 'detection_results.csv', 'Detection_Images')
    else:
        st.error("No character found. Please try again with a different image.")

else:
    st.warning("Please upload an image file.")
#st.text("A DIP project by Mukund Prasad H S")
# In[ ]:




