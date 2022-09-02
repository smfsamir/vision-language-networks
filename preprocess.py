
import os
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

root_dir = '../input/caltech256/256_ObjectCategories'
# get all the folder paths
all_paths = os.listdir(root_dir)

# create a DataFrame
data = pd.DataFrame()

images = []
labels = []
counter = 0
for folder_path in tqdm(all_paths, total=len(all_paths)):
    # get all the image names in the particular folder
    image_paths = os.listdir(f"{root_dir}/{folder_path}")
    # get the folder as label
    label = folder_path.split('.')[-1]
    
    if label == 'clutter':
        continue

    # save image paths in the DataFrame
    for image_path in image_paths:
        if image_path.split('.')[-1] == 'jpg':
            data.loc[counter, 'image_path'] = f"{root_dir}/{folder_path}/{image_path}"
            labels.append(label)
            counter += 1

labels = np.array(labels)
# one-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# add the image labels to the dataframe
for i in range(len(labels)):
    index = np.argmax(labels[i])
    data.loc[i, 'target'] = int(index)
    
# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

print(f"Number of labels or classes: {len(lb.classes_)}")
print(f"The first one hot encoded labels: {labels[0]}")
print(f"Mapping the first one hot encoded label to its category: {lb.classes_[0]}")
print(f"Total instances: {len(data)}")
 
# save as CSV file
data.to_csv('data.csv', index=False)
 
print(data.head(5))
