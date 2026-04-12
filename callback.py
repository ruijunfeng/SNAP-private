import os
from lightning.pytorch.callbacks import Callback

from utils.json_utils import save_json
from utils.plot_utils import parse_events_and_save_plot


class ResultCheckpoint(Callback):
    def __init__(
        self, 
        log_dir: str = None, 
    ):
        super().__init__()
        self.log_dir = log_dir
    
    # ================= Stage Setup (Crucial for log_dir) =================
    def setup(self, trainer, pl_module, stage):
        # This hook runs right before fit/test starts. 
        # By this time, trainer.logger.log_dir is guaranteed to be fully resolved.
        if self.log_dir is None: # None means is training from scratch, otherwise is loading from checkpoint
            if trainer.logger is not None:
                self.log_dir = trainer.logger.log_dir
            else:
                self.log_dir = trainer.default_root_dir
    
    # ================= Fit Cycle (Training + Validation for Early Stopping) =================
    def on_fit_end(self, trainer, pl_module):
        # Visualize the TensorBoard logs
        for filename in os.listdir(self.log_dir):
            if filename.startswith("events.out.tfevents"):
                parse_events_and_save_plot(
                    event_file_path=f"{self.log_dir}/{filename}",
                    output_image_path=f"{self.log_dir}/pictures.png",
                )
    
    # ================= Test Cycle (Final Evaluation) =================
    def on_test_start(self, trainer, pl_module):
        self.predictions = {}
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        idx = outputs["index"]
        self.predictions[idx] = {
            "y_pred": outputs["y_pred"],
            "y_proba": outputs["y_proba"],
            "logits": outputs["logits"],
        }
    
    def on_test_end(self, trainer, pl_module):
        save_json(f"{self.log_dir}/predictions.json", self.predictions)
