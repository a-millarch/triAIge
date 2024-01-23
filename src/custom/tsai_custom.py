from tsai.all import TensorMultiCategory, Categorize, BCEWithLogitsLossFlat

class CustomTSMultiLabelClassification(Categorize):
    "Reversible combined transform of multi-category strings to one-hot encoded `vocab` id"
    loss_func,order=BCEWithLogitsLossFlat(),1
    def __init__(self, c=None, vocab=None, add_na=False, sort=True): 
        super().__init__(vocab=vocab,add_na=add_na,sort=sort)
        self.c = c

    def setups(self, dsets):
        if not dsets: return


    def encodes(self, o):
        return TensorMultiCategory(o)


from matplotlib import pyplot as plt
from fastai.callback.core import Callback
from fastcore.basics import store_attr, range_of
import numpy as np

class TrainingShowGraph(Callback):
    "(Modified) Update a graph of training and validation loss"
    order,run_valid=65,False
    names = ['train', 'valid']
    def __init__(self, plot_metrics:bool=True, final_losses:bool=True, perc:float=.5):
        store_attr()

    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")
        if not(self.run): return
        self.nb_batches = []
        self.learn.recorder.loss_idxs = [i for i,n in enumerate(self.learn.recorder.metric_names[1:-1]) if 'loss' in n]
        _metrics_info = [(i,n) for i,n in enumerate(self.learn.recorder.metric_names[1:-1]) if 'loss' not in n]

        if len(_metrics_info) > 0: 
            self.metrics_idxs, self.metrics_names = list(zip(*_metrics_info))
        else: 
            self.metrics_idxs, self.metrics_names = None, None

    def after_train(self): self.nb_batches.append(self.train_iter - 1)


    def after_epoch(self):
 
        "Plot validation loss in the pbar graph"
        if not self.nb_batches: return
        rec = self.learn.recorder
        if self.epoch == 0:
            self.rec_start = len(rec.losses)
        iters = range_of(rec.losses)
        all_losses = rec.losses if self.epoch == 0 else rec.losses[self.rec_start-1:]

        modified_recorder_values = [sublist[:-1] for sublist in self.learn.recorder.values]
        
        val_losses = np.stack(modified_recorder_values)[:, self.learn.recorder.loss_idxs[-1]].tolist()
        if rec.valid_metrics and val_losses[0] is not None:
            all_losses = all_losses + val_losses
        else:
            val_losses = [None] * len(iters)
        y_min, y_max = min(all_losses), max(all_losses)
        margin = (y_max - y_min) * .05
        x_bounds = (0, len(rec.losses) - 1)
        y_bounds = (y_min - margin, y_max + margin)
        self.update_graph([(iters, rec.losses), (self.nb_batches, val_losses)], x_bounds, y_bounds)

    def after_fit(self):
        if hasattr(self, 'graph_ax'):
            plt.close(self.graph_ax.figure)
        if self.plot_metrics: 
            self.learn.plot_metrics(final_losses=self.final_losses, perc=self.perc)

    def update_graph(self, graphs, x_bounds=None, y_bounds=None, figsize=(6,4)):
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)
        self.graph_ax.clear()
        if len(self.names) < len(graphs): self.names += [''] * (len(graphs) - len(self.names))
        for g,n in zip(graphs,self.names): 
            if (g[1] == [None] * len(g[1])): continue
            self.graph_ax.plot(*g, label=n)
        self.graph_ax.legend(loc='upper right')
        self.graph_ax.grid(color='gainsboro', linewidth=.5)
        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        self.graph_ax.set_title(f'Losses\nepoch: {self.epoch +1}/{self.n_epoch}')
        self.graph_out.update(self.graph_ax.figure)