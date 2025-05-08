# PyTorch Workflow Library

A lightweight framework to streamline experiment management, checkpointing, and logging for PyTorch training projects.

## Features

* **Abstract Trainer Interface**: Define your model, data loaders, training loop, and evaluation once by subclassing `Trainer`.
* **Workflow Orchestration**: Automatically handles:

  * Experiment directory creation
  * Saving and loading of hyperparameters (`args.json`)
  * Checkpointing model weights and optimizer state (`last_checkpoint.pth`, `last_weights.pt`)
  * Resuming training from the last checkpoint
  * Per-epoch timing and metric logging to CSV (`history.csv`)
* **Flexible Internals**: Your `Trainer` can name its model and optimizer attributes as you like. The framework interacts via `get_model()` and `get_optimizer()` methods, so you’re not constrained to fixed attribute names.

## Installation

Clone this repository into your project directory:

```bash
git clone https://github.com/yourusername/pytorch-workflow.git
cd pytorch-workflow
```

Install requirements:

```bash
pip install torch torchvision scikit-learn tqdm
```



## Overview of Key Steps

Below, each block corresponds to the methods you implement in your `Trainer` subclass, plus how `Workflow` orchestrates them.

### 1. `setup()` block

**Purpose**: Build every component that depends on your hyperparameters (`args`).

```python
def setup(self):
    """
    - Instantiate `self.model` (a `torch.nn.Module`).
    - Create `self.optimizer` (a `torch.optim.Optimizer`).
    - Build any data loaders, loss functions, schedulers, etc.
    """
    # Example:
    self.model = MyModel(...).to(device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])
    self.loss_fn = nn.CrossEntropyLoss()
    train_ds = MyDataset(...)
    self.train_loader = DataLoader(train_ds, batch_size=self.args['batch_size'], shuffle=True)
    # Similarly for validation/test loader...
```

* Called once at the start of a **new run**.
* Called again after loading `args.json` during **resume**, before loading checkpoint.

### 2. `train_loop()` block

**Purpose**: Execute one epoch of training.

```python
def train_loop(self) -> dict:
    """
    - Runs over `self.train_loader` for one epoch.
    - Updates model weights via `self.optimizer`.
    - Returns a metrics dictionary, e.g.: `{'train_loss': avg_loss}`.
    """
    total_loss = 0
    for x, y in self.train_loader:
        out = self.model(x)
        loss = self.loss_fn(out, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
    return {'train_loss': total_loss / len(self.train_loader)}
```

* **Must return** a Python `dict` of metrics.
* These returned values are logged to `history.csv` along with a timestamp for training duration.

### 3. `evaluate()` block

**Purpose**: Run validation or test to measure performance.

```python
def evaluate(self) -> dict:
    """
    - Runs over validation/test loader.
    - Computes metrics (accuracy, F1, etc.).
    - Returns a metrics dictionary, e.g.: `{'val_acc': accuracy}`.
    """
    correct, total = 0, 0
    for x, y in self.val_loader:
        with torch.no_grad():
            out = self.model(x)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return {'val_acc': correct / total}
```

* **Must return** a `dict` of evaluation metrics.
* These metrics and the evaluation duration are also appended to `history.csv`.

### 4. `Workflow` orchestration

`Workflow` ties everything together:

1. **Initialization**: `Workflow(trainer, path, epochs, name=None)` stores your `Trainer` instance, base `path`, total `epochs`, and optional experiment `name`.

2. **Run** (`workflow.run()`):

   * Checks if `path/name` exists:

     * **New**: creates directory, saves `args.json`, calls `trainer.setup()`.
     * **Resume**: loads `args.json` into `trainer.args`, calls `trainer.setup()`, loads last checkpoint & weights.
   * Verifies that `trainer.model`, `trainer.optimizer`, and `trainer.args` are set.

3. **Training Loop** (`run_training`):

   * Loops from start epoch to `epochs`:

     1. Times and calls `trainer.train_loop()`.
     2. Saves checkpoint (`last_checkpoint.pth`) and weights (`last_weights.pt`).
     3. Times and calls `trainer.evaluate()`.
     4. Appends a row to `history.csv` containing:

        * `epoch` index
        * metrics from `train_loop` and `evaluate`
        * training and evaluation durations

---

## Quick Usage Example

1. **Define your trainer** (`mnist_trainer.py`):

   ```python
   from train import Trainer
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader

   class MnistTrain(Trainer):
       def __init__(self, args):
           self.args = args

       def setup(self):
           # build model, optimizer, loaders...
           pass

       def train_loop(self):
           # train one epoch
           return {'train_loss': 0.5}

       def evaluate(self):
           # validate/test
           return {'val_acc': 0.9}
   ```

2. **Main script** (`main.py`):

   ```python
   from workflow import Workflow
   from mnist_trainer import MnistTrain

   args = {'learning_rate':1e-3, 'batch_size':64, ...}
   trainer = MnistTrain(args)

   wf = Workflow(trainer, path='experiments', epochs=10, name='mnist_test')
   wf.run()
   ```

3. **Results**:

   * `experiments/mnist_test/args.json`
   * `last_checkpoint.pth`, `last_weights.pt`
   * `history.csv` with per-epoch metrics and timings

---

Now each block clearly explains its role and return requirements—logging everything you need for experiment tracking.


#### Constructor

```python
Workflow(
    trainer: Trainer,
    path: str,
    epochs: int,
    name: Optional[str] = None
)
```

* `trainer`: instance of your `Trainer` subclass
* `path`: base directory for experiments
* `epochs`: total number of training epochs
* `name`: optional experiment name (auto-generated if `None`)

#### Methods

* `run()`: create or resume the experiment, handle checkpointing, and loop through training + evaluation.

## Contributing

Feel free to open issues or pull requests for:

* Scheduler / callback integration
* TensorBoard or MLflow logging
* Multi-GPU / distributed training support

---

Happy training! 🚀
