from config import configurations
from train import train_process

if __name__ == '__main__':
  data_path = sys.argv[1]
  config = configurations(data_path)
  print ('config:\n', vars(config))
  train_losses, val_losses, bleu_scores = train_process(config)
