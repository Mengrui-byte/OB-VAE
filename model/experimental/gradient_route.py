"""Forward-identical quantity-gradient routing controls for a shared latent VAE."""
from model.autoencoder.unified_book_vae import UnifiedOrderBookVAE


class GradientRouteOrderBookVAE(UnifiedOrderBookVAE):
    def __init__(self, *, detach_quantity_latent=False, detach_quantity_geometry=False, **kwargs):
        super().__init__(**kwargs)
        self.gradient_route_config = dict(
            detach_quantity_latent=bool(detach_quantity_latent),
            detach_quantity_geometry=bool(detach_quantity_geometry),
        )

    def decode(self, z, mask, tick_norm, times):
        if z.shape != (mask.shape[0],64,self.latent_channels):
            raise ValueError('Unique whole-window latent shape does not match configuration')
        price = self.price_decoder(z,mask,tick_norm,times)
        # detach shares values/storage; this does not draw a new latent.
        zq = z.detach() if self.gradient_route_config['detach_quantity_latent'] else z
        geometry = (
            {key:value.detach() for key,value in price.items()}
            if self.gradient_route_config['detach_quantity_geometry'] else price
        )
        volume = self.volume_decoder(zq,geometry,mask,tick_norm,times)
        return dict(**price,volume=volume)
