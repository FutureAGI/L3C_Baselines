import argparse
import numpy
import matplotlib.pyplot as plt

def plot_mean_variance(ax, x, y_mean, y_var, color, lab):
    ax.plot(x, y_mean, '-', color=color, label=lab)
    ax.fill_between(x, y_mean - y_var, y_mean + y_var, color=color, alpha=0.15)

def plot_input_file(ax, file_name, color, lab, max_step):
    x = numpy.arange(max_step)
    y_mean = []
    y_var = []
    with open(file_name, "r") as fin:
        for line in fin.readlines():
            toks = line.strip().split('\t')
            if(len(toks) < 2):
                var = 0.01
            else:
                var = float(toks[1])
            y_mean.append(float(toks[0]))
            y_var.append(var)
    y_mean = numpy.asarray(y_mean)
    y_var = numpy.asarray(y_var)
    ms = min(y_mean.shape[0], max_step)
    plot_mean_variance(ax, x[:ms], y_mean[:ms], y_var[:ms], color, lab)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--max_time_step', type=int, default=1920)
    parser.add_argument('--y_axis', type=str, default=None)
    parser.add_argument('--input_info', type=str, help="File1,Color1,Lab1;File2,Color2,Lab2; ...")
    args = parser.parse_args()

    fig, ax = plt.subplots()
    if(args.y_axis is not None):
        y_tok = map(float(args.y_axis.split(",")))
        plt.ylim(y_tok[0], y_tok[1])

    input_info = args.input_info.split(";")
    for inputs in input_info:
        input_tok = inputs.split(",")
        plot_input_file(ax, input_tok[0], input_tok[1], input_tok[2], args.max_time_step)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(args.output_path)
