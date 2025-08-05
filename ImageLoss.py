import torch
import torch.nn as nn
import torch.nn.functional as F


def huber( x, delta ):
    x = x.abs()
    return (x.square()/(2*delta)).clip(None, delta/2) + F.relu( x - delta )

def spatial_diff( x ):
    xh = x[:,:,:,1:] - x[:,:,:,:-1]
    xv = x[:,:,1:,:] - x[:,:,:-1,:]
    return xh, xv

def channel_diff( x ):
    return x - x.roll( shifts=1, dims=1)

class ImageLoss(nn.Module):
    def __init__(self, alpha=10, beta=5, delta=5/255):
        super(ImageLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward( self, x, y ):
        loss = huber(x-y, self.delta).mean( dim=(1,2,3) )

        if( self.alpha > 0 ):
            xh, xv = spatial_diff( x )
            yh, yv = spatial_diff( y )
            loss = loss + self.alpha * ( huber(xh-yh, self.delta).mean( dim=(1,2,3) ) + huber(xv-yv, self.delta).mean( dim=(1,2,3) ) )

        if( self.beta > 0 ):
            xc = channel_diff( x )
            yc = channel_diff( y )
            loss = loss + self.beta * huber( xc-yc, self.delta).mean( dim=(1,2,3) )

        return loss

if( __name__ == "__main__" ):
    x = torch.randn( 8, 3, 32, 32 )
    y = torch.randn( 8, 3, 32, 32 )

    cri = ImageLoss()

    loss = cri( x, y ).mean()

    print( loss )

