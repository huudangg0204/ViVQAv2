import matplotlib

#load image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('./data/vivqa_v2/images/536725.jpg')

#load features npy file
import numpy as np
features = np.load('./features/vinvl_vinvl/536725.npy', allow_pickle=True)
# get objects, boxes, and attributes in features
objects = features.item().get('objects')
boxes = features.item().get('boxes')
attributes = features.item().get('attributes')

dict_objects = [{
    'object': obj,
    'box': box,
    'attributes': attr
} for obj, box, attr in zip(objects, boxes, attributes)]

#plot image
plt.imshow(img)

#plot objects
for obj in dict_objects:
    box = obj['box']
    plt.gca().add_patch(matplotlib.patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none'))
    plt.text(box[0], box[1], obj['object'], fontsize=5, color='black')
    plt.text(box[0], box[1] + 10, obj['attributes'], fontsize=5, color='black')
plt.axis('off')
plt.show()


