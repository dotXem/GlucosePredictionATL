import sys
import datetime


def printd(*msg):
    """
        Enhanced print function that prints the date and time of the log.
        :param msg: messages
        :return: /
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(date, " ".join(str(v) for v in msg))
    sys.stdout.flush()
