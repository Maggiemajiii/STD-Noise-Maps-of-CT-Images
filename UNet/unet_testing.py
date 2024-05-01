#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TESTING

model.eval() # set in evaluation mode

test_loss = 0.0
test = []
test_image = []
ground_truth = []
predicted_patches = []

model.to(device)

with torch.no_grad():
    for batch in test_loader:
        # moving data to device
        images = batch['ct_generated'].float().to(device)
        std_maps = batch['std_map'].float().to(device)
        
        images = images.unsqueeze(0)
        std_maps = std_maps.unsqueeze(0)

        y_pred = model(images) # predicting using the model

        # calculating loss
        loss = average_relative_error(y_pred, std_maps)
        test_loss += loss.item() * images.size(0)
        test.append(test_loss)
        y_pred = y_pred.unsqueeze(0)

        # appending data for visualization
        test_image.append(images.cpu().numpy())
        ground_truth.append(std_maps.cpu().numpy())
        predicted_patches.append(y_pred.cpu().numpy())

# calculating average test loss
test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")


# In[ ]:





# In[ ]:




