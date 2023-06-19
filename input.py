from scipy.io import mmread
from scipy.sparse import csr_matrix
import torch
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import uuid

import click
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from ptdec.dec import DEC
from ptdec.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy

mtx = mmread('AF.mtx')
csr = csr_matrix(mtx)
csr_tensor = torch.sparse_csr_tensor(torch.tensor(csr.indptr, dtype=torch.int64),
                                     torch.tensor(csr.indices, dtype=torch.int64),
                                     torch.tensor(csr.data), dtype=torch.double)



class SparseTensorDataset(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        self.csr_tensor = csr_tensor
        self.cuda = cuda
        self.testing_mode = testing_mode #default FALSE
        self._cache = dict()

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = list(self.csr_tensor[idx])
            if self.cuda:
                self._cache[idx][0] = self._cache[idx][0].cuda(non_blocking=True)
                self._cache[idx][1] = torch.tensor(
                    self._cache[idx][1], dtype=torch.long
                ).cuda(non_blocking=True)
        return self.csr_tensor[idx]


    def __len__(self):
        return 128 if self.testing_mode else len(self.csr_tensor)

#csr_tensor = csr_tensor
dataset = SparseTensorDataset([csr_tensor], cuda=True)
dataloader = DataLoader(dataset, batch_size=128)

@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=True
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=300,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=500,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    csr_tensor_train = SparseTensorDataset(
        train=True, cuda=cuda, testing_mode=testing_mode
    )  # training dataset
    csr_tensor_val = SparseTensorDataset(
        train=False, cuda=cuda, testing_mode=testing_mode
    )  # evaluation dataset
    autoencoder = StackedDenoisingAutoEncoder(
        [199507 * 6880, 500, 500, 2000, 5], final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    print("Pretraining stage.")
    ae.pretrain(
        csr_tensor_train,
        autoencoder,
        cuda=cuda,
        validation=csr_tensor_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2,
    )
    print("Training stage.")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(
        csr_tensor_train,
        autoencoder,
        cuda=cuda,
        validation=csr_tensor_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback,
    )
    print("DEC stage.")
    model = DEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=csr_tensor_train,
        model=model,
        epochs=100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda,
    )
    predicted, actual = predict(
        csr_tensor_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)
    if not testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = (
            confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
        )
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion).get_figure().savefig(
            "confusion_%s.png" % confusion_id
        )
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
        writer.close()


if __name__ == "__main__":
    main()
