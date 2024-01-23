import paddle
import paddle.profiler as profiler


import paddle
import paddle.profiler as profiler

class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = paddle.rand(shape=[100], dtype='float32')
        label = paddle.randint(0, 10, shape=[1], dtype='int64')
        return image, label

    def __len__(self):
        return self.num_samples

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = paddle.nn.Linear(100, 10)

    def forward(self, image, label=None):
        return self.fc(image)

dataset = RandomDataset(20 * 4)
simple_net = SimpleNet()
opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=simple_net.parameters())
BATCH_SIZE = 4
loader = paddle.io.DataLoader(
dataset,
batch_size=BATCH_SIZE)
p = profiler.Profiler(targets=[profiler.ProfilerTarget.GPU],
                      profile_memory=True, timer_only=True)
p.start()
for i, (image, label) in enumerate(loader()):
    out = simple_net(image)
    loss = paddle.nn.functional.cross_entropy(out, label)
    avg_loss = paddle.mean(loss)
    avg_loss.backward()
    opt.minimize(avg_loss)
    simple_net.clear_gradients()
    p.step(num_samples=BATCH_SIZE)
    if i % 10 == 0:
         step_info = p.step_info(unit='images')
         print("Iter {}: {}".format(i, step_info))
         # The average statistics for 10 steps between the last and this call will be
         # printed when the "step_info" is called at 10 iteration intervals.
         # The values you get may be different from the following.
         # Iter 0:  reader_cost: 0.51946 s batch_cost: 0.66077 s ips: 6.054 images/s
         # Iter 10:  reader_cost: 0.00014 s batch_cost: 0.00441 s ips: 907.009 images/s
p.stop()
