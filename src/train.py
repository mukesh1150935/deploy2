#train.py

import torch
import time
import matplotlib.pyplot as plt

from data_pre import PrepareData
from setting import EPOCHS
from model import SimpleLossCompute, LabelSmoothing, NoamOpt
from model import make_model

from setting2 import TGT_VOCAB, SRC_VOCAB

from setting import LAYERS, D_MODEL, D_FF, DROPOUT, H_NUM,  \
    SAVE_FILE, TRAIN_FILE, DEV_FILE
    
    
def unfreeze(model):
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.generator.parameters():
        param.requires_grad = True
        
def eval(model, loss_fn, target_embeddings, target_uni_embeddings, target_ngram_embeddings, data, is_cov=False, pre_train=False):
    total_loss = 0
    total_words = 0
    total_other_loss = 0
    model.eval()
    for i in range(len(data)):
        batch = data[i][:-1] # exclude original indices
        outputs, fert = model(batch, is_cov, pre_train)
        if is_cov:
            targets = batch[3]
            targets_cov = targets.view(-1).long()
            outputs = outputs.view(-1, outputs.size(2))
            # gradOutput = loss_cov(outputs, targets_cov)
            # loss = gradOutput.data[0]
            other_loss = 0.0
        else:
            targets = batch[1][1:]  # exclude <s> from targets
            loss, _, other_loss = loss_fn(
                    outputs, targets, target_embeddings, target_uni_embeddings, target_ngram_embeddings, fert, model.generator, is_cov, True)
        # loss, _, other_loss = loss_fn(
        #         outputs, targets, target_embeddings, model.generator, opt, is_cov, eval=True)
        total_loss += loss
        total_other_loss += other_loss
        # total_words += targets.data.ne(onmt.Constants.PAD).float().sum()

    model.train()
    return total_loss / total_words, total_other_loss / total_words

def trainModel(model, trainData, validData, dataset, target_embeddings, target_uni_embeddings, target_ngram_embeddings, optim):
    def freeze():
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        for param in model.generator.parameters():
            param.requires_grad = False
        for param in model.encoder.cov_fc1.parameters():
            param.requires_grad = True
        for param in model.encoder.cov_fc2.parameters():
            param.requires_grad = True
        for param in model.encoder.cov_fc3.parameters():
            param.requires_grad = True
        # optim.set_lr(opt.cov_lr)
        optim.set_parameters(model.parameters())

    print(model)
    # sys.stdout.flush()
    model.train()
        

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
            print("Epoch no. %d Batch id: %d Loss: %f Tokens/sec: %fs" % (
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
        print('......... Evaluation')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        dev_losses.append(dev_loss)
        print(f'Epoch {epoch + 1} Evaluation Loss: {dev_loss}')
        
        print('<<<<< Evaluate loss: %f' % dev_loss)
        # Save the model if the current epoch's dev loss is better than the previous best loss and update the best loss
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), SAVE_FILE)
            best_dev_loss = dev_loss
            print('Saving Model.........')
        print()

    return train_losses, dev_losses

if __name__ == '__main__':
    print("Loading data.....")
    data = PrepareData(TRAIN_FILE, DEV_FILE)

    print('Model Training....')
    print('Building model...')
    train_start = time.time()
    # Loss function
    criterion = LabelSmoothing(TGT_VOCAB, padding_idx=0, smoothing=0.0)
    # Optimizer
    optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    train_losses, dev_losses = train(data, model, criterion, optimizer)
    print(f'<<< Training completed, time elapsed {time.time() - train_start:.4f} seconds')

    # Debug prints to ensure losses are captured
    # print('Train losses:', train_losses)
    # print('Dev losses:', dev_losses)

    # Plot loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(dev_losses, label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Plots: Training and Validation Over Each Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    # plt.show()

