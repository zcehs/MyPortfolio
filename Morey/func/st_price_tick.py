import math

def get_tick(price):

    if price < 10:
        return 0.01
    elif price < 50:
        return 0.05
    elif price < 100:
        return 0.1
    elif price < 500:
        return 0.5
    elif price < 1000:
        return 1
    else:
        return 5


def limit_up_price(prev_close):
    if prev_close == 0:
        return -1

    raw_price = prev_close * 1.1

    tick = get_tick(raw_price)

    return round(math.floor(raw_price / tick) * tick, 2)

def limit_down_price(prev_close):
    if prev_close == 0:
        return -1

    raw_price = prev_close * 0.9

    tick = get_tick(raw_price)

    return round(math.ceil(raw_price / tick) * tick, 2)

def limit_up_change(prev_close):
    if prev_close == 0:
        return 100

    change = (limit_up_price(prev_close) - prev_close) / prev_close * 100
    return change

def limit_down_change(prev_close):
    if prev_close == 0:
        return -100

    change = (limit_down_price(prev_close) - prev_close) / prev_close * 100
    return change