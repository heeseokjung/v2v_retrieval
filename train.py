from utils.main_utils import (
    init_config,
    init_dataloaders,
    init_model,
    init_trainer
)


def main():
    # init configuration
    cfg, logger = init_config("config/default.yaml")
    
    # init dataloader
    logger.info(f"init dataloader... [{cfg.DATASET.name}]")
    train_dataloader, val_dataloader = init_dataloaders(cfg)
    
    # init model
    logger.info(f"init model... args={cfg.MODEL}")
    model = init_model(cfg)
    
    # init trainer
    logger.info(
        f"init trainer... batch_size: {cfg.TRAIN.batch_size} lr: {cfg.TRAIN.lr} epochs: {cfg.TRAIN.num_epochs})"
    )
    trainer = init_trainer(cfg)
    
    # train
    logger.info(f"start training")
    trainer.fit(model, train_dataloader, val_dataloader)
    
            
if __name__ == "__main__":
    main()