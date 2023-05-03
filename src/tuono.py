import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from NNmodel import NN
from transformers import BertTokenizer
from dataloader import FullDataloader
from callbacks import PrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
profiler = PyTorchProfiler()
logger = TensorBoardLogger('tb_logs', name='Autextification_model')
model = NN(2, 1e-5)
train_data = FullDataloader('train', '/home/matteo/PycharmProjects/NLPassignment/src/cache', tokenizer,
                            '/home/matteo/PycharmProjects/NLPassignment/src/rawData', ['autextication'])
val_data = FullDataloader('validation', '/home/matteo/PycharmProjects/NLPassignment/src/cache', tokenizer,
                          train_data.corpora_path, train_data.corpus_list)
test_data = FullDataloader('test', '/home/matteo/PycharmProjects/NLPassignment/src/cache', tokenizer,
                           train_data.corpora_path, train_data.corpus_list)


train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=train_data.collate)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=val_data.collate)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=test_data.collate)

trainer = pl.Trainer(
    profiler="simple",
    logger=logger,
    max_epochs=1,
    callbacks=[PrintingCallback(), EarlyStopping(monitor = 'val loss')],
    fast_dev_run=True)
trainer.fit(model, train_loader, val_loader)
#trainer.validate(model, val_loader)
trainer.test(model, test_loader)
