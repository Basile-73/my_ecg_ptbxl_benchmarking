from trainer import SimpleTrainer
from pathlib import Path

def main():
    train_config = Path('configs/train_config.yaml')
    trainer = SimpleTrainer(train_config)
    trainer.train()

if __name__ == '__main__':
    main()
