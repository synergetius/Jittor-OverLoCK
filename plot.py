import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
matplotlib.use('Agg')
# LOG_NAME = 'log.csv'
LOG_NAME = 'log-2-ep20.csv'
log = pd.read_csv(LOG_NAME)
print(log)
print()
ep = len(log.train_loss)
plt.figure(figsize = (15, 10))

plt.subplot(2, 2, 1)
plt.plot(np.arange(1, ep + 1), list(log.train_loss), label='Train Loss')
plt.title('Train Loss'); plt.legend()
plt.xticks(np.arange(1, ep + 1))
# plt.figure(figsize = (15, 10))
plt.subplot(2, 2, 2)
plt.plot(np.arange(1, ep + 1), list(log.val_loss), label='Validation Loss')
plt.title('Validation Loss'); plt.legend()
plt.xticks(np.arange(1, ep + 1))

# # plt.subplot(2, 2, 2)
# # plt.plot(old_accs, label='Old Task Accuracy')
# # plt.plot(new_accs, label='New Task Accuracy')
# # plt.title('Training Accuracy'); plt.legend()

plt.subplot(2, 2, 3)
plt.plot(np.arange(1, ep + 1), list(log.acc_top1 * 100), label='Validation Accuracy')
plt.title('Validation Accuracy'); plt.legend()
plt.xticks(np.arange(1, ep + 1))
# plt.tight_layout()
# plt.savefig('training_metrics.png')
# plt.close()
plt.savefig('training_metrics.png')