import torch.nn.functional as F

def count_parametrs(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy_v2(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()

def accuracy(pred, label, epsilon=0.0001):
    count_profit = 0
    count_loss = 0
    predict = pred.detach().numpy()
    lbl = label.detach().numpy()
    for i in range(len(label)):
        accuracy_now_profit = (lbl[i][0]-predict[i][0])
        accuracy_now_loss = (lbl[i][1] - predict[i][1])
        if accuracy_now_profit < epsilon:
            count_profit += 1
        if accuracy_now_loss < epsilon:
            count_loss += 1
    return count_profit, count_loss

def accuracy_v3(pred, label, epsilon=0.0001):
    count = 0
    predict = pred.detach().numpy()
    lbl = label.detach().numpy()
    for i in range(len(label)):
        accuracy_now = abs(lbl[i]-predict[i])
        # print(accuracy_now)
        if accuracy_now < epsilon:
            count += 1
    # print(count)
    return count


def accuracy_v4(pred, label, epsilon=0.0001):
    count = 0
    predict = pred.detach().cpu().numpy()
    lbl = label.detach().numpy()
    for i in range(len(label)):
        accuracy_now = abs(lbl[i]-predict[i])
        # print(accuracy_now)
        if accuracy_now < epsilon:
            count += 1
    # print(count)
    return count


def TenArrToNP(arr):

    res = []
    # print(arr)
    for minArr in arr:
        # print(minArr.detach().numpy())
        res.extend(minArr.detach().numpy())

    for i in range(len(res)):
        res[i] = res[i][0]
    return res


def average(labels, results):
    avg_lbl = sum(labels) / len(labels)
    avg_res = sum(results) / len(results)

    print(f"Average for label_tensor is : {avg_lbl}\nAverage for result_tensor is : {avg_res}")

    return avg_lbl, avg_res


def calculated_standard_deviation(labels, results):
    if len(labels) != len(results):
        raise ValueError('The lists must have the same length for calculating standard deviation')

    n = len(labels)
    squared_diff_sum = 0
    for i in range(n):
        squared_diff_sum += (results[i] - labels[i]) ** 2

    mean_squared_diff = squared_diff_sum/n
    standard_deviation = (mean_squared_diff)**0.5

    print(f'Standard deviation is : {standard_deviation}')
    return standard_deviation


def calculated_error(standard_deviation, avg_lbl):
    error = (standard_deviation/avg_lbl)*100
    print(f'Error is : {error}%')
    return error


def calculated_max_error(labels, results):

    if len(results) != len(labels):
        raise ValueError('The lists must have the same length for calculating max error')

    max_error = (abs(results[0] - labels[0])/labels[0]) * 100
    for i in range(1, len(results)):
        now = (abs(results[i] - labels[i])/labels[i]) * 100
        if now > max_error:
            max_error = now

    print(f'Max error is : {max_error}%')
    return max_error