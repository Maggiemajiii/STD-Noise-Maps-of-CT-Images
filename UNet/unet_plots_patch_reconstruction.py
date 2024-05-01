#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TRAINING-VALIDATION LOSS CURVES

plt.figure(figsize=(7, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.xticks(range(0, len(train_losses), 1))  # specifying tick locations 
plt.legend()
plt.show()


# In[ ]:


# PATCH RECONSTRUCTION
# visualizing prediction

# converting results to numpy arrays
test_image_array = np.array(test_image)
ground_truth_array = np.array(ground_truth)
predicted_patches_array = np.array(predicted_patches)

# reshaping the results to form an 8x8 grid to fit each patch
test_image_array = test_image_array.reshape(8,8,64,64)
ground_truth_array = ground_truth_array.reshape(8,8,64,64)
predicted_patches_array = predicted_patches_array.reshape(8,8,64,64)

# concatenating patches along the axis to reconstruct final image
test_image_final = np.concatenate([np.concatenate(row, axis=1) for row in test_image_array], axis=0)
ground_truth_final = np.concatenate([np.concatenate(row, axis=1) for row in ground_truth_array], axis=0)
predicted_patches_final = np.concatenate([np.concatenate(row, axis=1) for row in predicted_patches_array], axis=0)


# In[ ]:


# PLOTTING RESULTS

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(np.rot90(test_image_final), cmap='gray')
plt.title('Test Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(np.rot90(ground_truth_final), cmap='hot')
plt.title('Ground Truth Noise Map')
plt.axis('off')

# brightness factor (adjust as needed according to preference)
brightness_factor = 8.0

# creating a brighter version of the 'hot' colormap
hot_brighter = plt.cm.hot(np.linspace(0, brightness_factor, 256))
hot_brighter = ListedColormap(hot_brighter)

plt.subplot(133)
# Plot the image with the brighter colormap
plt.imshow(np.rot90(predicted_patches_final), cmap=hot_brighter)
plt.title('Predicted Noise Map')
plt.axis('off')
plt.show()

