import time


def calculate_diff_time(previous_time):
    '''
    :param previous_time:之前的时间
    :return: 两者的差值时间
    '''
    elapsed = time.time() - previous_time
    return elapsed
