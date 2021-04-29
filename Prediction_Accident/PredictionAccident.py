from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn import svm, tree
from sklearn.metrics import confusion_matrix
def read_data(path: str):
    return pd.read_csv(path)


def drop_info(drop_list: list, data: pd.DataFrame):
    for d_l in drop_list:
        data = data.drop(d_l, axis=1)
    return data


def make_d(_data: pd.DataFrame, attr: str) -> OrderedDict:
    data = OrderedDict()
    for d in _data[attr]:
        if data.get(d, None) == None:
            data[d] = 0
        else:
            data[d] += 1

    data_l = sorted(data.items(), key=lambda x: x[1])
    data = OrderedDict()
    for l in data_l:
        data[l[0]] = l[1]

    return data


#  feature normalize
def stand_norm(df):
    _max = df.max()
    _min = df.min()
    _mean = df.mean()
    return (df - _min) / (_max - _min)


def random_split_data(raw_data_x, raw_data_y):

    randomize = np.arange(len(raw_data_x))
    np.random.shuffle(randomize)
    random_data_x = raw_data_x[randomize]
    random_data_y = raw_data_y[randomize]
    split_size = int(len(random_data_y) * 0.8)
    train_x = random_data_x[:split_size]
    train_y = random_data_y[:split_size]
    dev_x = random_data_x[split_size:]
    dev_y = random_data_y[split_size:]
    return train_x, train_y, dev_x, dev_y


def nn(func, train_x, train_y, dev_x, dev_y):
    clf = func()
    clf = clf.fit(train_x, train_y)
    pred = clf.predict(dev_x)
    prediction = clf.predict(train_x)
    print(
        f"{func.__name__}\tdev acc: {np.sum(pred==dev_y)/len(dev_y)}\ttrain acc: {np.sum(prediction==train_y)/len(train_y)}"
    )

    # confusion matrix
    cm = confusion_matrix(y_true=dev_y, y_pred=pred)
    print(cm)


def main():
    # read data
    casualty_data = read_data(
        "../../data/kaggle/RoadSafetyData/dftRoadSafetyData_Casualties_2018.csv"
    )
    accident_data = read_data(
        "../../data/kaggle/RoadSafetyData/dftRoadSafetyData_Accidents_2018.csv"
    )

    # merage and drop nan
    casualty_and_accident_data = pd.merge(
        accident_data, casualty_data, on="Accident_Index"
    )
    casualty_and_accident_data = casualty_and_accident_data.drop(
        ["Accident_Index"], axis=1
    )
    casualty_and_accident_data = casualty_and_accident_data.dropna()

    # convert data type
    Date_d = make_d(casualty_and_accident_data, "Date")
    Time_d = make_d(casualty_and_accident_data, "Time")
    Highway_d = make_d(casualty_and_accident_data, "Local_Authority_(Highway)")
    LSOA_d = make_d(casualty_and_accident_data, "LSOA_of_Accident_Location")

    #  generate new feature and drop some features
    casualty_and_accident_data["Date_d"] = [
        Date_d[d] for d in casualty_and_accident_data["Date"]
    ]
    casualty_and_accident_data["Time_d"] = [
        Time_d[d] for d in casualty_and_accident_data["Time"]
    ]
    casualty_and_accident_data["Highway_d"] = [
        Highway_d[d] for d in casualty_and_accident_data["Local_Authority_(Highway)"]
    ]
    casualty_and_accident_data["LSOA_d"] = [
        LSOA_d[d] for d in casualty_and_accident_data["LSOA_of_Accident_Location"]
    ]
    casualty_and_accident_data = casualty_and_accident_data.drop(
        ["Date", "Time",
            "Local_Authority_(Highway)", "LSOA_of_Accident_Location"],
        axis=1,
    )

    # norm
    ret = pd.DataFrame()
    for f in casualty_and_accident_data:

        ret[f] = stand_norm(casualty_and_accident_data[f])

    # sort by target
    ret = ret.sort_values(by=["Casualty_Severity"])

    # split input abd target
    raw_data_x = np.array(ret.drop("Casualty_Severity", axis=1).values)
    raw_data_y = np.array(ret["Casualty_Severity"].values)

    # 0 ,0.5,1 -> 0, 1, 2
    raw_data_y = 2 * raw_data_y

    sam = {i: 0 for i in [0, 1, 2]}
    for y in raw_data_y:
        sam[y] += 1
    print(sam)

    # sample balance
    X = np.zeros((sam[0] * 3, raw_data_x.shape[1]))
    Y = np.zeros((sam[0] * 3,))
    a, b = sam[0], sam[0] + sam[1]
    for i, (x, y) in enumerate(
        zip(raw_data_x[: sam[0]], raw_data_y[: sam[0]])
    ):  # :sam[0]
        X[i] = x
        Y[i] = y
    for i, (x, y) in enumerate(
        zip(raw_data_x[sam[0]: sam[0] + sam[0]],
            raw_data_y[sam[0]: sam[0] + sam[0]])
    ):
        # sam[0]: sam[0]+sam[0]
        X[sam[0] + i] = x
        Y[sam[0] + i] = y

    for i, (x, y) in enumerate(
        zip(
            raw_data_x[sam[0] + sam[1]: sam[0] + sam[1] + sam[0]],
            raw_data_y[sam[0] + sam[1]: sam[0] + sam[1] + sam[0]],
        )
    ):
        X[sam[0] * 2 + i] = x
        Y[sam[0] * 2 + i] = y

    train_x, train_y, dev_x, dev_y = random_split_data(X, Y)
    print(train_x.shape, train_y.shape, dev_x.shape, dev_y.shape)

    # ML algorithm
    func_list = [tree.DecisionTreeClassifier, svm.SVC]
    kwargs = dict(
        train_x=train_x,
        train_y=train_y,
        dev_x=dev_x,
        dev_y=dev_y,
    )
    for fun in func_list:
        nn(fun, **kwargs)


if __name__ == "__main__":
    main()
