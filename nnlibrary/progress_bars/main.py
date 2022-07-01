from nnlibrary.progress_bars.abstractions import AbstractProgressBar


class SimpleProgressBar(AbstractProgressBar):
    def __call__(self, *args, **kwargs):
        pass

# import time
#
# # A List of Items
# items = list(range(0, 57))
# ln = len(items)
#
# # Initial call to print 0% progress
# print_progress_bar(0, ln, prefix='Progress:', suffix='Complete', length=50)
# for i, item in enumerate(items):
#     # Do stuff...
#     time.sleep(0.1)
#     # Update Progress Bar
#     print_progress_bar(i + 1, ln, prefix='Progress:', suffix='Complete', length=50)


import math


# Add class representation
def progress_bar(iteration: int,
                 total: int,
                 time_passed: float,
                 prefix: str = 'Progress:',
                 suffix: str = '',
                 decimals: int = 1,
                 length: int = 50,
                 fill: str = '#',
                 print_end: str = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        time_passed - Required  : total time passed (Float)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    percent = ("{0:5." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    minutes = int(time_passed // 60)
    seconds = int(math.floor(time_passed % 60))
    prettify_time = "{:02}:{:02}".format(minutes, seconds)

    print(f'\r{prefix} |{bar}| {percent}% {suffix} <{prettify_time}>', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
