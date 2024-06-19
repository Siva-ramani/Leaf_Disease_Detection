# Iterate over the training set and print out a few image filenames along with their labels
for images, labels in training_set:
    for i in range(len(labels)):
        # Get the filename of the i-th image
        filename = training_set.filepaths[i]
        # Get the corresponding label
        label = labels[i]
        print(f"Filename: {filename}, Label: {label}")
    break  # Print only the first batch of images for brevity

# Do the same for the validation set
for images, labels in valid_set:
    for i in range(len(labels)):
        filename = valid_set.filepaths[i]
        label = labels[i]
        print(f"Filename: {filename}, Label: {label}")
    break
