import pytorch_lightning as pl
from torch import nn, optim
import torchmetrics
import transformers
from stats import*



class NN(pl.LightningModule):
    def __init__(self,num_classes, learning_rate):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.classification_head = nn.Linear(768, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(task='binary', num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task='binary', num_classes=num_classes, average='macro')

    def forward(self, input_encodings):
        input_ids, attention_mask = input_encodings.input_ids, input_encodings.attention_mask
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classification_head(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(preds, labels)
        f1_score = self.f1(preds, labels)
        self.log_dict({'train_accuracy': accuracy, 'train_f1': f1_score}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _common_step(self, batch, batch_idx):
        input_encodings, labels = batch
        logits = self.forward(input_encodings)
        loss = self.loss_fn(logits, labels)
        print(logits.shape)
        print(logits)
        return loss, logits, labels
    '''
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    '''

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
