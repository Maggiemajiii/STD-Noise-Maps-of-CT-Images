#!/usr/bin/env python
# coding: utf-8

# In[1]:


# initializing the model and optimizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(n_channels = 1, n_classes = 1, dropout_rate=0.0).to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.0001)


# In[ ]:


# LOSS Function - Average Relative Error
# relative error followed by averaging
def average_relative_error(output, target):
    absolute_error = torch.abs(output - target)
    nonzero_mask = target != 0  # Mask for non-zero elements in target
    return torch.mean(torch.abs((absolute_error[nonzero_mask] / target[nonzero_mask]))) 


# In[ ]:


# training and validation

train_losses = []
val_losses = []
num_epochs = 300

for epoch in range(num_epochs):
    running_loss = 0.0
    running_val_loss = 0.0
    model.train()  # setting the model in training mode

    for idx, images in enumerate(tqdm(train_loader)):
        img = images['ct_generated'].float().to(device) # loading images
        img = img.unsqueeze(0)
        std_map = images['std_map'].float().to(device)
        std_map = std_map.unsqueeze(0)
        
        optimizer.zero_grad()
        y_pred = model(img) # predictions

        loss = average_relative_error(y_pred, std_map) # loss calculation
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss) # appending resulting loss

    # Validation phase
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for idx, image in enumerate(tqdm(val_loader)):
            img = image['ct_generated'].float().to(device) # loading images
            img = img.unsqueeze(0)
            std_map = image['std_map'].float().to(device)
            std_map = std_map.unsqueeze(0)

            y_pred = model(img) # predictions
            val_loss = average_relative_error(y_pred, std_map) # loss calculation
            running_val_loss += val_loss.item() * img.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss) # appending resulting loss

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

