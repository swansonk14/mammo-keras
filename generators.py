import numpy as np
import traceback

def batch_generator(metadata, batch_size, image_pipeline, label_pipeline, debug=False):
    batch_images = [None]*batch_size
    batch_labels = [None]*batch_size
    b_i = 0
    
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
