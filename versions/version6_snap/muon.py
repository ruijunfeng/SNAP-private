
    def training_step(self, batch, batch_idx):
        # 1. Retrieve the optimizers and learning rate schedulers
        opt_muon, opt_adamw = self.optimizers()
        sch_muon, sch_adamw = self.lr_schedulers()
        
        # 2. Forward pass
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            numeric_features=batch["numeric_features"],
            labels=batch["labels"],
        )
        loss = outputs["loss"]
        
        # 3. Manual backward pass
        self.manual_backward(loss)
        
        # 4. Update weights for both optimizers
        opt_muon.step()
        opt_adamw.step()
        
        # 5. Update learning rate schedulers 
        # (Assuming you want to step the scheduler every batch. If you want epoch-level, move this to on_train_epoch_end)
        sch_muon.step()
        sch_adamw.step()
        
        # 6. Clear gradients for the next step
        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        
        # 7. Log the loss
        self.log("total_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        
        # When using manual optimization, returning the loss is optional as Lightning won't use it
        return loss
    
    def configure_optimizers(self):
        # Set the log directory for saving predictions
        self.log_dir = self.trainer.log_dir
        
        muon_params = []
        adamw_params = []
        
        # 1. Separate parameters: 2D+ for Muon, 1D for AdamW
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)
                
        # 2. Initialize both optimizers
        opt_muon = torch.optim.Muon(muon_params, lr=self.hparams.config.lr)
        # Note: AdamW usually requires a smaller learning rate than Muon. Adjust the multiplier (e.g., 0.1) as needed.
        opt_adamw = torch.optim.AdamW(adamw_params, lr=self.hparams.config.lr * 0.1)
        
        # 3. Calculate total training steps for the schedulers
        total_steps = self.hparams.num_training_samples * self.trainer.max_epochs // self.hparams.config.batch_size // self.trainer.accumulate_grad_batches
        
        # 4. Initialize learning rate schedulers for both optimizers
        sch_muon = get_scheduler(
            name=self.hparams.config.scheduler_name, 
            optimizer=opt_muon, 
            num_warmup_steps=0, 
            num_training_steps=total_steps,
        )
        sch_adamw = get_scheduler(
            name=self.hparams.config.scheduler_name, 
            optimizer=opt_adamw, 
            num_warmup_steps=0, 
            num_training_steps=total_steps,
        )
        
        # 5. Return lists of optimizers and schedulers
        return [opt_muon, opt_adamw], [sch_muon, sch_adamw]
    