* trigger should decide the index level (consistent with grow_atom, i.e. which modules to grow), arch just duplicate that level that's all (thus only needs grow_atom)
* plus operation is tricky, duplicate which one?
    * maybe the middle one? (duplicate_times+1) // 2, each time add a 3-3-3

* may need smooth, temporacy current err dominate 

* this implementation trained significantly slower than original because needs to capture the activations (hooker)
    * If this is a must, wouldn't it means the training will always be slower if we use the training set as indicator
    * so turns out can only use validation set as indicator
