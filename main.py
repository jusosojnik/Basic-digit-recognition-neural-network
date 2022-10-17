from tkinter import *
from tkinter import simpledialog
import neural_network as nn
import numpy as np

canvas_width = 500
canvas_height = 450

symbol = []
numPat = 0
symbols = []
values = []
NN = nn.NeuralNetwork(0, 0, 0, 0, 0)


def draw_vector(symbol):
    for e in symbol:
        c.create_oval(e[0] - 2, e[1] - 2, e[0] + 2, e[1] + 2, fill='red', outline='red')


def normalize(symbol):
    max_x = max([_[0] for _ in symbol])
    min_x = min([_[0] for _ in symbol])
    max_y = max([_[1] for _ in symbol])
    min_y = min([_[1] for _ in symbol])

    scale = max([(max_x - min_x), (max_y - min_y)])
    y_main_axis = True if scale == (max_y - min_y) else False

    if y_main_axis:
        middle = (max_x - (max_x - min_x) / 2) - min_x
    else:
        middle = (max_y - (max_y - min_y) / 2) - min_y
    offset = 0.5 - middle / scale

    for i in range(len(symbol)):
        if y_main_axis:
            symbol[i] = ([((symbol[i][0] - min_x) / scale) + offset, (symbol[i][1] - min_y) / scale])
        else:
            symbol[i] = ([(symbol[i][0] - min_x) / scale, ((symbol[i][1] - min_y) / scale) + offset])

    return symbol


def vectorize(symbol, y):
    last_symbol = symbol[len(symbol) - 1]
    while True:
        for i in range(2, len(symbol), 2):
            if i >= len(symbol):
                break
            symbol[i] = [(symbol[i][0] + symbol[i - 1][0]) / 2, (symbol[i][1] + symbol[i - 1][1]) / 2]
            symbol.remove(symbol[i - 1])
            symbol[len(symbol) - 1] = last_symbol
            if len(symbol) == int(spin_boxVpp.get()):
                break
        if len(symbol) == int(spin_boxVpp.get()):
            break

    draw_vector(symbol)

    symbol = normalize(symbol)
    tmp_x = []
    tmp_y = []

    for i in symbol:
        tmp_x.append(i[0])
        tmp_y.append(i[1])
    tmp = tmp_x + tmp_y
    if y:
        symbols.append(tmp)
    else:
        return tmp


def print_symbol():
    global numPat
    vectorize(symbol, True)
    ans = simpledialog.askinteger("Symbol", "Enter drawn symbol")
    values.append(ans)
    c.delete("all")
    numPat += 1
    symbol.clear()
    numPattern.config(text="Number of patterns: {}".format(numPat))


def init_nn():
    global NN
    tmp = []
    for e in values:
        if e not in tmp:
            tmp.append(e)

    NN = nn.NeuralNetwork(int(spin_boxVpp.get()) * 2, int(spin_boxNohn.get()), len(tmp), float(spin_boxLr.get()), float(spin_boxEt.get()))


def train():
    tmp = []
    for e in values:
        if e not in tmp:
            tmp.append(e)
    tmp = np.array(tmp)
    tmp = np.sort(tmp)
    new_values = np.zeros((len(values), len(tmp)))
    for i, odgovor in enumerate(values):
        new_values[i][np.where(tmp == odgovor)[0]] = 1

    new_symbols = np.array(symbols)
    NN.train(new_symbols, new_values)


def clean():
    global symbol
    symbol = []
    c.delete("all")


def reset():
    global symbol, numPat, symbols, values
    symbol = []
    numPat = 0
    symbols = []
    values = []
    numPattern.config(text="Number of patterns: {}".format(numPat))
    c.delete("all")


def recognize():
    global symbol
    s = vectorize(symbol, False)
    od = NN.predict(s, False)
    tmp = []
    for e in values:
        if e not in tmp:
            tmp.append(e)
    tmp = np.array(tmp)
    tmp = np.sort(tmp)
    rec.config(text="Recognized: {}".format(tmp[np.where(od == np.max(od))][0]))


def paint(event):
    color = 'white'
    symbol.append([event.x, event.y])
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    c.create_oval(x1, y1, x2, y2, fill=color, outline=color)


root = Tk()
root.title("Neural Network")
root.geometry('500x800')

c = Canvas(root, height=canvas_height, width=canvas_width, bg="green")
c.pack(expand=YES, fill=BOTH)
c.bind('<B1-Motion>', paint)

###############################################################################
frameVpp = Frame(root)
frameVpp.pack(side=BOTTOM)

labelVpp = Label(frameVpp, text="Vectors per pattern")
labelVpp.pack(side=LEFT)

spin_boxVpp = Spinbox(frameVpp, from_=2, to=30)
spin_boxVpp.pack(side=LEFT)

###############################################################################
frameNohn = Frame(root)
frameNohn.pack(side=BOTTOM)

labelNohn = Label(frameNohn, text="Number of hidden neurons")
labelNohn.pack(side=LEFT)

spin_boxNohn = Spinbox(frameNohn, from_=1, to=30)
spin_boxNohn.pack(side=LEFT)

###############################################################################
frameLr = Frame(root)
frameLr.pack(side=BOTTOM)

labelLr = Label(frameLr, text="Learning rate")
labelLr.pack(side=LEFT)

spin_boxLr = Spinbox(frameLr, from_=0, to=1, format="%.2f", increment=0.01)
spin_boxLr.pack(side=LEFT)

###############################################################################
frameEt = Frame(root)
frameEt.pack(side=BOTTOM)

labelEt = Label(frameEt, text="Error tolerance")
labelEt.pack(side=LEFT)

spin_boxEt = Spinbox(frameEt, from_=0, to=1, format="%.3f", increment=0.001)
spin_boxEt.pack(side=LEFT)

##############################################################################
settings = Label(root, text="Settings")
settings.pack(side=BOTTOM)

store = Button(root, text="Store", command=print_symbol)
store.pack(side=BOTTOM)

initialize = Button(root, text="Initialize", command=init_nn)
initialize.pack(side=BOTTOM)

rst = Button(root, text="Reset", command=reset)
rst.pack(side=BOTTOM)

rcg = Button(root, text="Recognize", command=recognize)
rcg.pack(side=BOTTOM)

clear = Button(root, text="Clear", command=clean)
clear.pack(side=BOTTOM)

train = Button(root, text="Train", command=train)
train.pack(side=BOTTOM)

numPattern = Label(root, text="Number of patterns: 0")
numPattern.pack(side=BOTTOM)

rec = Label(root, text="Recognized:")
rec.pack(side=BOTTOM)

root.mainloop()
