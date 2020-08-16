from config import configurations
from test import test_process
import sys

# 在執行 Test 之前，請先行至 config 設定所要載入的模型位置
if __name__ == '__main__':
  data_path = sys.argv[1]
  output_path = sys.argv[2]
  config = configurations(data_path)
  print ('config:\n', vars(config))
  test_loss, bleu_score = test_process(config, output_path)
  print (f'test loss: {test_loss}, bleu_score: {bleu_score}')