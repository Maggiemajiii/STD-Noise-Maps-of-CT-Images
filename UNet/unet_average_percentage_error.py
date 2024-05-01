#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AVERAGE PERCENTAGE ERROR - APE

epsilon = 1e-8 # to avoid division by zero
ground_truth_final_safe = ground_truth_final + (ground_truth_final == 0) * epsilon

# calculating the percentage error
percent_error = (ground_truth_final - predicted_patches_final) / ground_truth_final_safe
average_percent_error = abs(np.mean(percent_error)) * 100  # Convert to percentage

print(f"Average Percentage Error: {average_percent_error:.2f}%")


# In[ ]:





# In[ ]:




