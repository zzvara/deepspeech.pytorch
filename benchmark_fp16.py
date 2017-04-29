# This script based on the parameters will benchmark the speed of F16 precision to F32 precision on GPU
import argparse
import json
import time

import torch
from torch.autograd import Variable
from tqdm import trange
from warpctc_pytorch import CTCLoss

from model import DeepSpeech, supported_rnns

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=8, help='Size of input')
parser.add_argument('--seconds', type=int, default=15,
                    help='The size of the fake input in seconds using default stride of 0.01, '
                         '15s is usually the maximum duration')
parser.add_argument('--dry-runs', type=int, default=2, help='Dry runs before measuring performance')
parser.add_argument('--runs', type=int, default=5, help='How many benchmark runs to measure performance')
parser.add_argument('--labels-path', default='labels.json', help='Path to the labels to infer over in the model')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
args = parser.parse_args()

input_standard = torch.randn(args.batch_size, 1, 161, args.seconds * 100).cuda()

rnn_type = args.rnn_type.lower()
assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

with open(args.labels_path) as label_file:
    labels = str(''.join(json.load(label_file)))

audio_conf = dict(sample_rate=args.sample_rate,
                  window_size=args.window_size)

model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                   nb_layers=args.hidden_layers,
                   audio_conf=audio_conf,
                   labels=labels,
                   rnn_type=supported_rnns[rnn_type])

print("Number of parameters: %d" % DeepSpeech.get_param_size(model))
parameters = model.parameters()
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4,
                            momentum=0.9, nesterov=True)
model = torch.nn.DataParallel(model).cuda()
criterion = CTCLoss()


def iteration(input_data, cuda_half=False):
    target = torch.IntTensor(int(args.batch_size * ((args.seconds * 100) / 2))).fill_(
        1)  # targets, align half of the audio
    target_size = torch.IntTensor(args.batch_size).fill_(int((args.seconds * 100) / 2))
    input_percentages = torch.IntTensor(args.batch_size).fill_(1)

    inputs = Variable(input_data)
    target_sizes = Variable(target_size)
    targets = Variable(target)
    start = time.time()
    fwd_time = time.time()
    out = model(inputs)
    out = out.transpose(0, 1)  # TxNxH
    torch.cuda.synchronize()
    fwd_time = time.time() - fwd_time

    seq_length = out.size(0)
    sizes = Variable(input_percentages.mul_(int(seq_length)).int())
    if cuda_half:
        out = out.cuda().float()
    loss = criterion(out, targets, sizes, target_sizes)
    loss = loss / inputs.size(0)  # average the loss by minibatch
    if cuda_half:
        loss = loss.cuda().half()
    bwd_time = time.time()
    # compute gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    bwd_time = time.time() - bwd_time
    end = time.time()
    return start, end, fwd_time, bwd_time


def run_benchmark(input_data, cuda_half=False):
    for n in trange(args.dry_runs):
        iteration(input_data, cuda_half)
    print('\nDry runs finished, running benchmark')
    running_time, total_fwd_time, total_bwd_time = 0, 0, 0
    for n in trange(args.runs):
        start, end, fwd_time, bwd_time = iteration(input_data, cuda_half)
        running_time += end - start
        total_fwd_time += fwd_time
        total_bwd_time += bwd_time
    bwd_time = total_bwd_time / float(args.runs)
    fwd_time = total_fwd_time / float(args.runs)
    return running_time / float(args.runs), fwd_time, bwd_time


print("Running standard benchmark")
run_time, fwd_time, bwd_time = run_benchmark(input_standard)

input_half = input_standard.cuda().half()
model = model.cuda().half()
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4,
                            momentum=0.9, nesterov=True)
print("\nRunning half precision benchmark")
run_time_half, fwd_time_half, bwd_time_half = run_benchmark(input_half, cuda_half=True)

print('\n')
print("Average times for DeepSpeech training in seconds: ")
print("F32 precision: Average training loop %.2fs Forward: %.2fs Backward: %.2fs " % (
    run_time, fwd_time, bwd_time))
print("F16 precision: Average training loop %.2fs Forward: %.2fs Backward: %.2fs " % (
    run_time_half, fwd_time_half, bwd_time_half))
