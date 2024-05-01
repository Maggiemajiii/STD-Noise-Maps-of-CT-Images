#!/usr/bin/env python
# coding: utf-8

# In[1]:


class DoubleConv(nn.Module): # 2d convolutional process
    
    def __init__(self, in_channels, out_channels, mid_channels = None, dropout_rate=0.0):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels # takes output channels as mid channels unless mentioned otherwise
            
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # 2d convolution
            nn.BatchNorm2d(mid_channels), # batch normalization
            nn.ReLU(inplace=True), # ReLU activation
            nn.Dropout2d(dropout_rate), # dropout (optional)
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
    def forward(self, x):
        return self.conv_op(x) # forward propagation
    
    
class Down(nn.Module): # downsampling (encoder) layer
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # max pooling
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

    
class Up(nn.Module): # upsampling (decoder) layer
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input format is CxHxW
        diffY = x2.size()[2] - x1.size()[2] # ensuring dimension uniformity
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # padding, ensuring dimension uniformity

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
    
class OutConv(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# In[ ]:


class UNet(nn.Module):
    
    ''' UNet class to call the model. Initialize with number of in channels and number of classes.
    Dropout is optional. The default is zero and can be changed as per preference. '''
    
    def __init__(self, n_channels, n_classes, bilinear=False, dropout_rate=0.0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate

        self.inc = (DoubleConv(n_channels, 64, dropout_rate=dropout_rate))
        self.down1 = (Down(64, 128)) # downsampling
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor)) # bottleneck
        self.up1 = (Up(1024, 512 // factor, bilinear)) # upsampling
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x): # forward propagation
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 1024
        x = self.up1(x5, x4) # 1024
        x = self.up2(x, x3) # 512
        x = self.up3(x, x2) # 256
        x = self.up4(x, x1) # 128
        logits = self.outc(x) # 64
        return logits

