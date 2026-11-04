from func.st_price_tick import limit_up_price, limit_down_price, limit_up_change, limit_down_change

def stock_color(prev, cur):
    background_color = "transparent"
    font_color = "black"

    if cur >= limit_up_price(prev):
        background_color = "red"
        font_color = "white"
    elif cur > prev:
        font_color = "red"
    elif cur == prev:
        font_color = "grey"
    elif cur < prev:
        font_color = "green"
    elif cur <= limit_down_price(prev):
        background_color = "green"
        font_color = "white"

    return background_color, font_color

def only_stock_color(prev, cur):

    if cur > prev:
        return "transparent", "red"
    elif cur == prev:
        return "transparent", "grey"
    elif cur < prev:
        return "transparent", "green"


'''def stock_color_for_change(change, prev_close):
    background_color = "transparent"
    font_color = "black"

    if change >= limit_up_change(prev_close):
        background_color = "red"
        font_color = "white"
    elif change > 0:
        font_color = "red"
    elif change == 0:
        font_color = "grey"
    elif change < 0:
        font_color = "green"
    elif change <= limit_down_change(prev_close):
        background_color = "green"
        font_color = "white"

    return background_color, font_color'''

def stock_color_for_change(change):

    background_color = "transparent"
    font_color = "black"
    if change > 0:
        font_color = "red"
    elif change == 0:
        font_color = "grey"
    elif change < 0:
        font_color = "green"

    return background_color, font_color

def stock_info_color(prev, cur, change):

    background_color = []

    font_color = []

    background_color.append(stock_color(prev.Close, cur.Close)[0])
    font_color.append(stock_color(prev.Close, cur.Close)[1])

    background_color.append(stock_color_for_change(change)[0])
    font_color.append(stock_color_for_change(change)[1])

    background_color.append(only_stock_color(prev.Close, cur.Open)[0])
    font_color.append(only_stock_color(prev.Close, cur.Open)[1])

    background_color.append(only_stock_color(prev.Close, cur.High)[0])
    font_color.append(only_stock_color(prev.Close, cur.High)[1])

    background_color.append(only_stock_color(prev.Close, cur.Low)[0])
    font_color.append(only_stock_color(prev.Close, cur.Low)[1])



    return background_color, font_color

def change_sign(change):

    if change > 0:
        return "+"
    elif change == 0:
        return ""
    else:
        return ""



