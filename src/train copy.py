#train.py

import torch
import time
import matplotlib.pyplot as plt

from data_pre import PrepareData
from setting import EPOCHS
from model import SimpleLossCompute, LabelSmoothing, NoamOpt
from model import make_model

from setting import LAYERS, D_MODEL, D_FF, DROPOUT, H_NUM, TGT_VOCAB, SRC_VOCAB, \
    SAVE_FILE, TRAIN_FILE, DEV_FILE

# Model initialization
model = make_model(
    SRC_VOCAB,
    TGT_VOCAB,
    LAYERS,
    D_MODEL,
    D_FF,
    H_NUM,
    DROPOUT
)

def run_epoch(data, model, loss_compute, epoch):
    """
    Iterate once over the dataset
    :param data: Dataset
    :param model: Model
    :param loss_compute: loss_compute function
    :param epoch: Current epoch number
    :return: Average loss
    """
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens  # Actual number of tokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (
                epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return (total_loss / total_tokens).item()  # Convert the tensor to a float value

def train(data, model, criterion, optimizer):
    """
    Train and save the model
    """
    # Initialize the best loss on the dev set to a large value
    best_dev_loss = 1e5

    train_losses = []
    dev_losses = []

    for epoch in range(EPOCHS):  # EPOCHS specified at the start of the script
        """
        Validate loss on the dev set after each epoch
        """
        model.train()
        train_loss = run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        train_losses.append(train_loss)
        print(f'Epoch {epoch + 1} Training Loss: {train_loss}')
        
        model.eval()

        # Evaluate loss on the dev set
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        dev_losses.append(dev_loss)
        print(f'Epoch {epoch + 1} Evaluation Loss: {dev_loss}')
        
        print('<<<<< Evaluate loss: %f' % dev_loss)
        # Save the model if the current epoch's dev loss is better than the previous best loss and update the best loss
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), SAVE_FILE)
            best_dev_loss = dev_loss
            print('****** Save model done... ******')
        print()

    return train_losses, dev_losses

if __name__ == '__main__':
    print('Processing data')
    data = PrepareData(TRAIN_FILE, DEV_FILE)

    print('>>> Start training')
    train_start = time.time()
    # Loss function
    criterion = LabelSmoothing(TGT_VOCAB, padding_idx=0, smoothing=0.0)
    # Optimizer
    optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    train_losses, dev_losses = train(data, model, criterion, optimizer)
    print(f'<<< Training completed, time elapsed {time.time() - train_start:.4f} seconds')

    # Debug prints to ensure losses are captured
    print('Train losses:', train_losses)
    print('Dev losses:', dev_losses)

    # Plot loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(dev_losses, label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    # plt.show()

