import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('Agg')

log = pd.read_csv('log.csv')
print(log)
print()

plt.figure(figsize = (15, 10))
plt.subplot(2, 2, 1)
plt.plot(list(log.train_loss), label='Train Loss')
plt.title('Train Loss'); plt.legend()

# plt.figure(figsize = (15, 10))
plt.subplot(2, 2, 2)
plt.plot(list(log.val_loss), label='Validation Loss')
plt.title('Validation Loss'); plt.legend()


# # plt.subplot(2, 2, 2)
# # plt.plot(old_accs, label='Old Task Accuracy')
# # plt.plot(new_accs, label='New Task Accuracy')
# # plt.title('Training Accuracy'); plt.legend()

plt.subplot(2, 2, 3)
plt.plot(list(log.acc_top1 * 100), label='Validation Accuracy')
plt.title('Validation Accuracy'); plt.legend()

# plt.tight_layout()
# plt.savefig('training_metrics.png')
# plt.close()
plt.savefig('training_metrics.png')