import numpy as np
import sys
import traceback

def batch_generator(split_group, metadata, batch_size=128, image_pipeline=None, label_pipeline=None, debug=False):

    metadata = [row for row in metadata if row.get('split_group', None) == split_group]
    b_i = 0
    batch_images = [None]*batch_size
    batch_labels = [None]*batch_size
    
    while True:
        np.random.shuffle(metadata)
        for row in metadata:
            try:
                batch_images[b_i] = image_pipeline.process(row)
                batch_labels[b_i] = label_pipeline.process(row)
                b_i += 1
                if b_i >= batch_size:
                    b_i = 0
                    batch_images = np.array(batch_images)
                    batch_labels = np.array(batch_labels)
                    yield batch_images, batch_labels
            except Exception as e:
                if debug:
                    traceback.print_exc()
                    continue
