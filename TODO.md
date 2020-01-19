* trigger should decide the index level (consistent with grow_atom, i.e. which modules to grow), arch just duplicate that level that's all (thus only needs grow_atom)
* plus operation is tricky, duplicate which one?
    * maybe the middle one? (duplicate_times+1) // 2, each time add a 3-3-3

* may need smooth, temporacy current err dominate 

* this implementation trained significantly slower than original because needs to capture the activations (hooker)
    * If this is a must, wouldn't it means the training will always be slower if we use the training set as indicator
    * so turns out can only use validation set as indicator

* pc2 issue when pcs2 is off, because we didn't include 'pc2' in trace in the sub-level
Epoch: [1 | 5] LR: 0.100000 Train-Loss: 2.3542 Val-Loss: 9.6014 Train-Acc: 14.0625 Val-Acc: 13.2000
    Total params: 0.27M
Traceback (most recent call last):
  File "cifar.py", line 838, in <module>
    main()
  File "cifar.py", line 561, in main
    errs = hooker.output(epoch)
  File "/home/chengyu/pytorch-classification/utils/hooker.py", line 303, in output
    record = layerHooker.output()
  File "/home/chengyu/pytorch-classification/utils/hooker.py", line 167, in output
    self.pca = self.records['act_pc2'].pca
KeyError: 'act_pc2'
[] [4] [Path]: checkpoints/cifar10/Debug-Batch-Orig-resnet-20-fixed-2-4-duplicate-constant-lr=0-1-noscale
