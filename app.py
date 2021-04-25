import time
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import ast

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

STOP_CURRENT_ITER_FLAG = False

def make_stop_current_inter_flag_true():
    global STOP_CURRENT_ITER_FLAG
    STOP_CURRENT_ITER_FLAG = True

class Perceptron(object):
    def __init__(self, no_of_inputs, minx, max_x, xa, xb, ya, yb, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        self.minx = minx
        self.max_x = max_x
        self.xa = xa
        self.xb = xb
        self.ya = ya
        self.yb = yb

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def abline(self, slope, intercept, fig, ax, num):
        """Plot a line from slope and intercept
        :param num:
        :param fig:
        :param ax:
        """

        x_vals = np.arange(self.minx, self.max_x, 0.1)
        # w1x + w2y + b
        # Ax + By - C = 0
        # y = (-(b / w2) / (b / w1)) <---   slope*x +  ntercept ---> (-b / w2)
        y_vals = intercept + slope * x_vals

        fig.tight_layout()
        ax.clear()
        ax.plot(self.xa, self.ya, 'x')
        ax.plot(self.xb, self.yb, 'o')
        ax.legend(labels=('cluster a', 'cluster b'), loc='best')  # legend placed at lower right
        ax.plot(x_vals, y_vals, '--')

        ax.set_title("my clustring perceptron")
        ax.set_xlabel('feature 1')
        ax.set_ylabel('feature 2')
        ax.axis('equal')
        ax.grid()
        ax.text(1, 0.7, f'epoch : {num}\n'
                        f'w1,w2=[{np.round_(self.weights[1],5)},{np.round_(self.weights[2],5)}]\n'
                        f'bias={np.round_(self.weights[0],5)}',
                style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(1)

    def plot_line(self, inputs, fig, ax, num):
        w1 = self.weights[1]
        w2 = self.weights[2]
        b = self.weights[0]
        # y_intercept
        y = -b / w2
        # slope --->
        m = -(b / w2) / (b / w1)
        self.abline(slope=m, intercept=y, fig=fig, ax=ax, num=num)

    def plot_data(self, inputs, targets, weights):
        # fig config
        plt.figure(figsize=(10, 6))
        plt.grid(True)

        # plot input samples(2D data points) and i have two classes.
        # one is +1 and second one is -1, so it red color for +1 and blue color for -1
        for input, target in zip(inputs, targets):
            plt.plot(input[0], input[1], 'ro' if (target == 1.0) else 'bo')

        # Here i am calculating slope and intercept with given three weights
        for i in np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1])):
            slope = -(weights[0] / weights[2]) / (weights[0] / weights[1])
            intercept = -weights[0] / weights[2]
            # y =mx+c, m is slope and c is intercept
            y = (slope * i) + intercept
            plt.plot(i, y, 'ko')

    def train(self, training_inputs, labels, fig, ax):
        global STOP_CURRENT_ITER_FLAG
        for epoch_num in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
            if STOP_CURRENT_ITER_FLAG == False:
                self.plot_line(training_inputs, fig, ax, epoch_num)
            else:
                STOP_CURRENT_ITER_FLAG = False
                break


class Hebb(object):
    # w new i = w old i + xi*y
    # bias new = bias old + y   ( bias is w0 or theta )
    def __init__(self, no_of_inputs, minx, max_x, xa, xb, ya, yb, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1, dtype=float)
        self.minx = minx
        self.max_x = max_x
        self.xa = xa
        self.xb = xb
        self.ya = ya
        self.yb = yb

    def abline(self, slope, intercept, fig, ax, epoch_num):
        """Plot a line from slope and intercept
        :param epoch_num:
        :param fig:
        :param ax:
        """

        x_vals = np.arange(self.minx, self.max_x, 0.1)
        # w1x + w2y + b
        # Ax + By - C = 0
        # y = (-(b / w2) / (b / w1)) <---   slope*x +  ntercept ---> (-b / w2)
        y_vals = intercept + slope * x_vals

        fig.tight_layout()
        ax.clear()
        ax.plot(self.xa, self.ya, 'x')
        ax.plot(self.xb, self.yb, 'o')
        ax.legend(labels=('cluster a', 'cluster b'), loc='best')  # legend placed at lower right
        ax.plot(x_vals, y_vals, '--')

        ax.set_title("my clustring perceptron")
        ax.set_xlabel('feature 1')
        ax.set_ylabel('feature 2')
        ax.axis('equal')
        ax.grid()
        ax.text(1, 0.7, f'epoch : {epoch_num}\n'
                        f'w1,w2=[{np.round_(self.weights[1],5)},{np.round_(self.weights[2],5)}]\n'
                        f'bias={np.round_(self.weights[0],5)}',
                style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(1)


    def plot_line(self, fig, ax, epoch_num):
        w1 = self.weights[2]
        w2 = self.weights[1]
        b = self.weights[0]

        # y_intercept
        y = -b / w2
        # slope --->
        m = -(b / w2) / (b / w1)
        self.abline(slope=m, intercept=y, fig=fig, ax=ax, epoch_num=epoch_num)

    def train(self, training_inputs, labels, fig, ax):
        global STOP_CURRENT_ITER_FLAG
        for epoch_num in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                # w new = w old + xi*y
                # b new  = b old + y
                # i tried both ways of using -1 and xnor but none of them works right
                # means decision boundary is not correct
                self.weights[1:] = self.weights[1:] + (
                        self.learning_rate * (np.invert(np.bitwise_xor(inputs.astype(int), label))))
                self.weights[0] += label
                # for i in range(3):
                #     if self.weights[i] == 0:
                #         self.weights[i] = -1
            if STOP_CURRENT_ITER_FLAG == False:
                self.plot_line(fig=fig, ax=ax, epoch_num=epoch_num)
            else:
                STOP_CURRENT_ITER_FLAG = False
                break


def merge(list1, list2):
    # merged_list = [np.array([list1[i], list2[i]]) for i in range(0, len(list1))]
    merged_list = []
    for i in range(0, len(list1)):
        merged_list.append(np.array([list1[i], list2[i]]))
    return merged_list


def get_numpy_array(a: str):
    return np.array(ast.literal_eval(re.sub(r'\]\s*\[',
                                            r'],[',
                                            re.sub(r'(\d+)\s+(\d+)',
                                                   r'\1,\2',
                                                   a.replace('\n', '')))))


def calculate_and_plot_hebb(number_of_samples_text_answer_A,
                            mean_vector_answer_A,
                            covariance_answer_A,
                            number_of_samples_text_answer_B,
                            mean_vector_answer_B,
                            covariance_answer_B,
                            learning_rate_answer,
                            epoch_number_answer,
                            fig,
                            ax):
    print("hebb btn works")

    training_inputs = []
    mean_a = get_numpy_array(mean_vector_answer_A).tolist()
    cov_a = get_numpy_array(covariance_answer_A).tolist()
    mean_b = get_numpy_array(mean_vector_answer_B).tolist()
    cov_b = get_numpy_array(covariance_answer_B).tolist()

    xa, ya = np.random.multivariate_normal(mean_a, cov_a, int(number_of_samples_text_answer_A)).T
    xb, yb = np.random.multivariate_normal(mean_b, cov_b, int(number_of_samples_text_answer_B)).T

    all_data_lst = []
    tmp_data_a = merge(xa, ya)
    tmp_data_b = merge(xb, yb)

    training_inputs = tmp_data_a + tmp_data_b

    all_labels = [0 for i in tmp_data_a]
    for i in tmp_data_b:
        all_labels.append(1)

    labels = np.array(all_labels)

    minxa = np.min(xa)
    minxb = np.min(xb)
    minx = 0
    if minxa > minxb:
        minx = minxb
    else:
        minx = minxa

    max_xb = np.max(xb)
    max_xa = np.max(xa)
    max_x = 0
    if max_xa > max_xb:
        max_x = max_xa
    else:
        max_x = max_xb

    my_hebb = Hebb(2,
                   threshold=int(epoch_number_answer),
                   learning_rate=float(learning_rate_answer),
                   minx=minx,
                   max_x=max_x,
                   xa=xa,
                   xb=xb,
                   ya=ya,
                   yb=yb)
    my_hebb.train(training_inputs, labels, fig=fig, ax=ax)


def calculate_and_plot_perceptron(number_of_samples_text_answer_A,
                                  mean_vector_answer_A,
                                  covariance_answer_A,
                                  number_of_samples_text_answer_B,
                                  mean_vector_answer_B,
                                  covariance_answer_B,
                                  learning_rate_answer,
                                  epoch_number_answer,
                                  fig,
                                  ax):
    print("perceptron btn works")
    training_inputs = []
    mean_a = get_numpy_array(mean_vector_answer_A).tolist()
    cov_a = get_numpy_array(covariance_answer_A).tolist()
    mean_b = get_numpy_array(mean_vector_answer_B).tolist()
    cov_b = get_numpy_array(covariance_answer_B).tolist()

    xa, ya = np.random.multivariate_normal(mean_a, cov_a, int(number_of_samples_text_answer_A)).T
    xb, yb = np.random.multivariate_normal(mean_b, cov_b, int(number_of_samples_text_answer_B)).T

    all_data_lst = []
    tmp_data_a = merge(xa, ya)
    tmp_data_b = merge(xb, yb)

    training_inputs = tmp_data_a + tmp_data_b

    all_labels = [0 for i in tmp_data_a]
    for i in tmp_data_b:
        all_labels.append(1)

    labels = np.array(all_labels)

    minxa = np.min(xa)
    minxb = np.min(xb)
    minx = 0
    if minxa > minxb:
        minx = minxb
    else:
        minx = minxa

    max_xb = np.max(xb)
    max_xa = np.max(xa)
    max_x = 0
    if max_xa > max_xb:
        max_x = max_xa
    else:
        max_x = max_xb
    my_perceptron = Perceptron(2, threshold=int(epoch_number_answer),
                               learning_rate=float(learning_rate_answer),
                               minx=minx,
                               max_x=max_x,
                               xa=xa,
                               xb=xb,
                               ya=ya,
                               yb=yb)
    my_perceptron.train(training_inputs, labels, fig=fig, ax=ax)


def calculate_and_plot_digram(number_of_samples_text_answer_A,
                              mean_vector_answer_A,
                              covariance_answer_A,
                              number_of_samples_text_answer_B,
                              mean_vector_answer_B,
                              covariance_answer_B,
                              fig,
                              ax
                              ):
    print("calculate_and_plot_digram  works")

    mean_a = get_numpy_array(mean_vector_answer_A).tolist()
    cov_a = get_numpy_array(covariance_answer_A).tolist()
    mean_b = get_numpy_array(mean_vector_answer_B).tolist()
    cov_b = get_numpy_array(covariance_answer_B).tolist()

    xa, ya = np.random.multivariate_normal(mean_a, cov_a, int(number_of_samples_text_answer_A)).T
    xb, yb = np.random.multivariate_normal(mean_b, cov_b, int(number_of_samples_text_answer_B)).T

    ax.clear()
    groupa = ax.plot(xa, ya, 'x')
    groupb = ax.plot(xb, yb, 'o')
    ax.axis('equal')
    ax.grid()
    fig.canvas.draw()
    # plt.plot(xa, ya, 'x')
    # plt.plot(xb, yb, 'o')
    # plt.axis('equal')
    # plt.show()


root = tk.Tk()

fig = plt.figure(figsize=(6, 4))
ax = fig.add_axes([0, 0, 1, 1])
plt.tight_layout()

############# All wigdets config
ax.legend(labels=('cluster a', 'cluster b'), loc='best')  # legend placed at lower right
ax.set_title("my clustring")
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()
canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

warning_label = tk.Label(root,
                         text="warning input your data like examples and click on buttons after!!! answering questions and do not click on buttons rapidly wait for response",
                         fg='red')
epoch_number = tk.Label(root, text="how much epoch : (input example --> 100)")
learning_rate = tk.Label(root, text="learning rate : (input example --> 0.01)")
epoch_number_answer = tk.Entry(root)  # .insert(0, "100")
learning_rate_answer = tk.Entry(root)  # .insert(0, "0.01")
#################
number_of_samples_text_question_A = tk.Label(root, text="how much sample (group A) : (input example --> 65)")
mean_vector_question_A = tk.Label(root, text="mean vector (group A) :    (input example --> [3,4] )")
covariance_question_A = tk.Label(root, text="covariance (group A) :   (input example -->    [[1, 0], [0, 100]]   )")
#################
number_of_samples_text_question_B = tk.Label(root, text="how much sample (group B) : (input example  -->  78)")
mean_vector_question_B = tk.Label(root, text="mean vector (group B) : (input example --> [33, 45]  )")
covariance_question_B = tk.Label(root, text="covariance (group B) :  (input example --> [[3, 6], [8, 100]]     )")
#################
number_of_samples_text_answer_A = tk.Entry(root)  # .insert(0, "65")
mean_vector_answer_A = tk.Entry(root)  # .insert(0, "[3,4]")
covariance_answer_A = tk.Entry(root)  # .insert(0, "[[1,0],[0,100]]")
number_of_samples_text_answer_B = tk.Entry(root)  # .insert(0, "78")
mean_vector_answer_B = tk.Entry(root)  # .insert(0, "[33,45]")
covariance_answer_B = tk.Entry(root)  # .insert(0, "[[3,6], [8,100]]")


stop_interation_btn = tk.Button(root, text="stop current epochs iteration", command=make_stop_current_inter_flag_true, fg='red')

################# calculations and plotting ##################
calculate_btn = tk.Button(root, text="plot data only!", command=lambda: calculate_and_plot_digram(
    number_of_samples_text_answer_A.get(),
    mean_vector_answer_A.get(),
    covariance_answer_A.get(),
    number_of_samples_text_answer_B.get(),
    mean_vector_answer_B.get(),
    covariance_answer_B.get(),
    fig=fig,
    ax=ax
))

hebb_calculator_btn = tk.Button(root, text="calculate and plot hebb unit"
                                , command=lambda: calculate_and_plot_hebb(
        number_of_samples_text_answer_A=number_of_samples_text_answer_A.get(),
        mean_vector_answer_A=mean_vector_answer_A.get(),
        covariance_answer_A=covariance_answer_A.get(),
        number_of_samples_text_answer_B=number_of_samples_text_answer_B.get(),
        mean_vector_answer_B=mean_vector_answer_B.get(),
        covariance_answer_B=covariance_answer_B.get(),
        learning_rate_answer=learning_rate_answer.get(),
        epoch_number_answer=epoch_number_answer.get(),
        fig=fig,
        ax=ax
    )
                                )
perceptron_calculator_btn = tk.Button(root, text="calculate and plot perceptron unit",
                                      command=lambda: calculate_and_plot_perceptron(
                                          number_of_samples_text_answer_A=number_of_samples_text_answer_A.get(),
                                          mean_vector_answer_A=mean_vector_answer_A.get(),
                                          covariance_answer_A=covariance_answer_A.get(),
                                          number_of_samples_text_answer_B=number_of_samples_text_answer_B.get(),
                                          mean_vector_answer_B=mean_vector_answer_B.get(),
                                          covariance_answer_B=covariance_answer_B.get(),
                                          learning_rate_answer=learning_rate_answer.get(),
                                          epoch_number_answer=epoch_number_answer.get(),
                                          fig=fig,
                                          ax=ax
                                      ))

######################## show widgets #############
# group a
warning_label.grid(column=0, row=17)
number_of_samples_text_question_A.grid(column=0, row=1)
number_of_samples_text_answer_A.grid(column=0, row=2)
mean_vector_question_A.grid(column=0, row=3)
mean_vector_answer_A.grid(column=0, row=4)
covariance_question_A.grid(column=0, row=5)
covariance_answer_A.grid(column=0, row=6)
# group b
number_of_samples_text_question_B.grid(column=0, row=7)
number_of_samples_text_answer_B.grid(column=0, row=8)
mean_vector_question_B.grid(column=0, row=9)
mean_vector_answer_B.grid(column=0, row=10)
covariance_question_B.grid(column=0, row=11)
covariance_answer_B.grid(column=0, row=12)
# learning rate and epoch number
epoch_number.grid(column=0, row=13)
epoch_number_answer.grid(column=0, row=14)
learning_rate.grid(column=0, row=15)
learning_rate_answer.grid(column=0, row=16)
#  buttons
calculate_btn.grid(column=0, row=18)
perceptron_calculator_btn.grid(column=0, row=19)
hebb_calculator_btn.grid(column=0, row=20)
stop_interation_btn.grid(column=0, row=21)
# matplotlib canvas
toolbar.grid(column=0, row=22)
canvas.get_tk_widget().grid(column=0, row=23)

####### Initiating answers for easy test
epoch_number_answer.insert(0, "100")
learning_rate_answer.insert(0, "0.01")
number_of_samples_text_answer_A.insert(0, "65")
mean_vector_answer_A.insert(0, "[3,4]")
covariance_answer_A.insert(0, "[[1,0],[0,100]]")
number_of_samples_text_answer_B.insert(0, "78")
mean_vector_answer_B.insert(0, "[33,45]")
covariance_answer_B.insert(0, "[[3,6], [8,100]]")

root.mainloop()
