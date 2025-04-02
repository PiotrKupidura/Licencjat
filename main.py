from core.model_torch_3 import Trainer


if __name__ == '__main__':

    trainer = Trainer(config="config.json")
    trainer.train()