from composer.models import ComposerModel
from torchmetrics import MetricCollection
from diffusion import Diffusion
from omegaconf import OmegaConf 

class ComposerDiffusionModel(ComposerModel):
    def __init__(self, model_config, collator_config, tokenizer):
        super().__init__()
        #TODO: define a tokenizer based on some collator config 
        #TODO: change the model config (make a new config file) to the diffusion config setup

        self.diffusion = Diffusion(model_config, tokenizer) #TODO: change tokenizer so diffusion can handle
        self._dummy_loss = lambda x,y: x["loss"]

        self.val_metrics = MetricCollection(self.diffusion.valid_metrics)
        self.train_metrics = MetricCollection(self.diffusion.train_metrics)

    def forward(self, batch):
        loss = self.diffusion._compute_loss(batch, prefix='train')
        return {'loss': loss}
    
    def eval_forward(self, batch, outputs=None):
        if outputs is not None:
            return outputs

        loss = self.diffusion._compute_loss(batch, prefix='val')

    def loss(self, outputs, batch):
        return outputs['loss']

    def update_metric(self, batch, outputs, metric):
        metric.update(outputs['loss'])

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def flops_per_batch(self, batch):
        return 0  
        
