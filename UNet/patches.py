#!/usr/bin/env python
# coding: utf-8

# In[1]:


def patches(image):
    ''' generating non-overlapping patches of size 64x64 '''
    demo_dict = []
    image = image.squeeze()
    patches = patchify(image.numpy(), (64,64), step=64)
        
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch_img = patches[i,j,:,:]
            demo_dict.append(single_patch_img)
    return demo_dict


# In[ ]:




