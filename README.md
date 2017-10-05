# Stochastic Sort
If we can get an array in order, then surely we can also get our lives in order.

## Training a Neural Network to Sort Arrays

### Architecture
The `Sorter` object utilizes two `Selector` networks.  
Each `Selector` selects an element in the target array, and the `Sorter` swaps the two elements.

The `Selector` networks have a single hidden ReLU layer—with 200 nodes by default—and a sigmoid-activated output layer.

### Sample Results
Here are the results of ten consecutive trials of 50000 iterations each.  
For each trial, a new `Sorter` is instantiated and a new array of ten random one-digit integers is generated.  
(This actually makes no sense because it has to re-learn how to sort with every trial; I'll fix it later)

`[7 3 6 7 7 5 1 3 9 4]` -> `[1 3 5 3 6 4 7 7 7 9]`, score diff `+80`

`[2 8 1 0 7 0 5 1 4 5]` -> `[4 0 1 5 1 2 7 8 5 0]`, score diff `+20`

`[8 2 3 1 5 2 1 8 2 4]` -> `[1 2 1 2 2 3 4 5 8 8]`, score diff `+80`

`[5 0 2 5 8 1 7 2 4 9]` -> `[5 2 0 2 9 5 0 8 7 4]`, score diff `-12`

`[8 3 9 1 5 3 4 1 0 8]` -> `[0 0 0 3 9 3 8 4 5 8]`, score diff `+72`

`[3 6 8 3 6 2 6 1 4 6]` -> `[3 2 3 1 4 6 6 6 6 8]`, score diff `+68`

`[0 9 9 2 8 6 4 7 6 1]` -> `[0 8 1 4 6 6 7 9 2 9]`, score diff `+60`

`[0 6 8 6 9 5 5 8 0 1]` -> `[0 0 5 5 8 1 6 8 9 6]`, score diff `+68`

`[8 7 3 8 2 8 8 3 3 4]` -> `[8 3 2 3 8 3 4 8 8 7]`, score diff `+40`

`[1 7 7 8 2 5 1 7 4 8]` -> `[1 2 1 4 5 7 7 7 8 8]`, score diff `+64`
