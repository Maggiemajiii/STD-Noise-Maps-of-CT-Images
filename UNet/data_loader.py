#!/usr/bin/env python
# coding: utf-8

# In[1]:


train_dataset, val_dataset, test_dataset = custom_dataset[:568], custom_dataset[568:640], custom_dataset[640:]
batch_size=1

#train_dataset, testval = train_test_split(custom_dataset, test_size=0.2, random_state=42)
#val_dataset, test_dataset = train_test_split(testval, test_size=0.5, random_state=42)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
print(len(train_loader), len(val_loader), len(test_loader))


# In[ ]:




