import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
matplotlib.use('Agg')
LOG_NAME = 'log.csv'
LOG_TORCH_NAME = 'log_torch.csv'
AUX_LOSS_RATIO = 0.4
# LOG_NAME = 'log-2-ep20.csv'

# plan of plot
# 训练过程：训练的整体loss，loss_main和loss_aux需要分别画曲线图
# 验证结果：验证loss（与loss_main对应），top-1 acc和top-5 acc分别画曲线图
# 考虑把val loss 和loss_main放在一张图上，因为有对照价值
log = pd.read_csv(LOG_NAME)
log_torch = pd.read_csv(LOG_TORCH_NAME)

log["train_loss_main"] = log.train_loss - log.train_loss_aux * AUX_LOSS_RATIO
log_torch["train_loss_main"] = log_torch.train_loss - log_torch.train_loss_aux * AUX_LOSS_RATIO
print()
ep = min(len(log.train_loss), len(log_torch.train_loss))
plt.figure(figsize = (20, 10))

plt.subplot(2, 3, 1)
plt.plot(np.arange(1, ep + 1), list(log.train_loss), '-', color = 'g', label='Jittor Train')
plt.plot(np.arange(1, ep + 1), list(log_torch.train_loss), '-', color = 'b', label='Torch Train')
plt.title('Total Train Loss'); plt.legend()
plt.xticks(np.arange(1, ep + 1))

plt.subplot(2, 3, 2)
plt.plot(np.arange(1, ep + 1), list(log.train_loss_main), '-', color = 'g', label='Jittor Train')
plt.plot(np.arange(1, ep + 1), list(log_torch.train_loss_main), '-', color = 'b', label='Torch Train')
plt.plot(np.arange(1, ep + 1), list(log.val_loss), '-.', color = 'g', label='Jittor Validation')
plt.plot(np.arange(1, ep + 1), list(log_torch.val_loss), '-.', color = 'b', label='Torch Validation')
# plt.plot(np.arange(1, ep + 1), list(log.val_loss), label='Validation Loss')
plt.title('Main Loss'); plt.legend()
plt.xticks(np.arange(1, ep + 1))

# # plt.subplot(2, 2, 2)
# # plt.plot(old_accs, label='Old Task Accuracy')
# # plt.plot(new_accs, label='New Task Accuracy')
# # plt.title('Training Accuracy'); plt.legend()
plt.subplot(2, 3, 3)
plt.plot(np.arange(1, ep + 1), list(log.train_loss_aux), '-', color = 'g', label='Jittor Train')
plt.plot(np.arange(1, ep + 1), list(log_torch.train_loss_aux), '-', color = 'b', label='Torch Train')
plt.title('Auxiliary Loss'); plt.legend()
plt.xticks(np.arange(1, ep + 1))


plt.subplot(2, 3, 4)
plt.plot(np.arange(1, ep + 1), list(log.acc_top1 * 100), '-', color = 'g', label='Jittor Validation')
plt.plot(np.arange(1, ep + 1), list(log_torch.acc_top1 * 100), '-', color = 'b', label='Torch Validation')
plt.title('Top-1 Accuracy'); plt.legend()
plt.xticks(np.arange(1, ep + 1))

plt.subplot(2, 3, 5)
plt.plot(np.arange(1, ep + 1), list(log.acc_top5 * 100), '-', color = 'g', label='Jittor Validation')
plt.plot(np.arange(1, ep + 1), list(log_torch.acc_top5 * 100), '-', color = 'b', label='Torch Validation')
plt.title('Top-5 Accuracy'); plt.legend()
plt.xticks(np.arange(1, ep + 1))
# plt.tight_layout()
# plt.savefig('training_metrics.png')
# plt.close()
plt.savefig('training_metrics.png')