import torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

class SleepStagePredictor:
    def __init__(self, config):
        self.config = config
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        if self.config.MODEL == "moirai":
            return MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{self.config.SIZE}"),
                prediction_length=self.config.PREDICTION_LENGTH,
                context_length=self.config.CONTEXT_LENGTH,
                patch_size=self.config.PATCH_SIZE,
                num_samples=self.config.NUM_SAMPLES,
                target_dim=1,
            )
        elif self.config.MODEL == "moirai-moe":
            return MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{self.config.SIZE}"),
                prediction_length=self.config.PREDICTION_LENGTH,
                context_length=self.config.CONTEXT_LENGTH,
                patch_size=16,
                num_samples=self.config.NUM_SAMPLES,
                target_dim=1,
            )
        
    def create_predictor(self):
        return self.model.create_predictor(batch_size=self.config.BATCH_SIZE)
