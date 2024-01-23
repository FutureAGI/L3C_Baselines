# Example1: train on default dynamic graph mode
import paddle
import numpy as np

# train on default dynamic graph mode
linear = paddle.nn.Linear(10, 10)
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=True)
sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
for epoch in range(20):
    for batch_id in range(5):
        x = paddle.uniform([10, 10])
        out = linear(x)
        loss = paddle.mean(out)
        loss.backward()
        sgd.step()
        sgd.clear_gradients()
        scheduler.step()    # If you update learning rate each step
        #print(sgd.get_lr())
    # scheduler.step()        # If you update learning rate each epoch

