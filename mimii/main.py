import optax
import jax
from flax import nnx
import grain.python as grain
import orbax.checkpoint as ocp
from pathlib import Path
ckpt_dir = Path(Path.cwd() / './checkpoints')
from net.cnn_transformer import Transformer
from util.parser import *
from loss.loss import LabelSmoothCrossEntropyLoss
from feeder.dataset import load_data

def loss_fn(model:Transformer, batch):
    _ , logits = model(batch['data'])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels=batch['label']).mean()
    return loss,logits

@nnx.jit
def train_step(model:Transformer,optimizer:nnx.Optimizer,metrics:nnx.MultiMetric,batch):
    grad_fn = nnx.value_and_grad(loss_fn,has_aux = True)
    (loss,logits),grads = grad_fn(model,batch)
    metrics.update(loss = loss,logits = logits,labels = batch['label'])
    optimizer.update(grads)

@nnx.jit
def eval_step(model:Transformer,metrics:nnx.MultiMetric,batch):
    loss,logits = loss_fn(model,batch)
    metrics.update(loss = loss,logits = logits,label = batch['label'])

@nnx.jit 
def pred_step(model:Transformer,batch):
    _,logits = model(batch['data'])
    return logits.argmax(axis = 1)

if __name__ =='__main__':
    batch_size = 256
    train_loader,test_loader = load_data('snr',num_workers = 8,num_epoch = 200,batch_size = batch_size)
    model = Transformer(rngs = nnx.Rngs(0),num_classes = 3)
    optimizer = nnx.Optimizer(model,optax.adamw(lr = 0.0001,b1 = 0.9))
    metrics = nnx.MultiMetric(
        accuracy = nnx.metrics.Accuracy(),
        loss  =nnx.metrics.Average('loss')
    )
    best_acc = 0.0
    train_steps = int(train_loader._sampler.__num_epochs *train_loader._sampler.num_recodes / 256)
    checkpointer = ocp.StandardCheckpointer()

    for step,batch in enumerate(train_loader):
        train_step(model,optimizer,metrics,batch)
        if step > 0 and (step % 500 ==0 or step == train_steps - 1):
            print("Step:{}_Train Acc@1: {} loss: {} ".format(step,metrics.compute()['accuracy'],metrics.compute()['loss']))
            metrics.reset()

            for test_batch in test_loader:
                eval_step(model,metrics,test_batch)
                print("Step:{}_Test Acc@1: {} loss: {} ".format(step,metrics.compute()['accuracy'],metrics.compute()['loss']))
            if metrics.compute()['accuracy'] > best_acc:
                best_acc = metrics.compute()['accuracy']
                _,state = nnx.split(model)
                checkpointer.save(ckpt_dir / 'best_snr',state)
            metrics.reset()