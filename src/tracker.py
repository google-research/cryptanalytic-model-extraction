from __future__ import print_function

import sys

from src.meta import Singleton


class Logger(metaclass=Singleton):
    """
    This replaces the default print behaviour in the program, to allow the
    programmer for more flexibility for printing specific statements.
    """
    #: Debug messages, for helping the programmers debug.
    DEBUG = 0
    #: Info messages, for displaying the state to the user.
    INFO = 1
    #: Debug messages, in case you want to not show any info messages.
    CRITICAL_DEBUG = 2
    #: Warning messages, for things that might not work, but do not otherwise influence execution.
    WARNING = 3
    #: Error messages, for logic that breaks the program.
    ERROR = 4

    def __init__(self, level=DEBUG, out=sys.stdout, err=sys.stderr):
        """
        Initialize a logger, note that as this is a singleton, it can only happen
        once, so the error level, out and err locations are relatively fixed,
        unless set manually.

        :param level:   The highest level that will be printed.
        :param out:     The place DEBUG and INFO messages will be printed to. Can be a stream or a file handle,
                        it currently does not support paths.
        :param err:     The place WARNING and ERROR messages will be printed to. Can be a stream or a file handle,
                        it currently does not support paths.
        """
        #: The highest level that will be printed, e.g. in case it is
        self.level = level
        #: DEBUG and INFO message location.
        self.out = out
        #: WARNING and ERROR message location.
        self.err = err

        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def log(self, *msg, level=DEBUG):
        """
        Print a specific message given a specific level, in case no level is specified,
        assume that it is a debug message.
        :param msg:     The message to be printed.
        :param level:   The level of the message (DEBUG, INFO, WARNING, ERROR).
        """
        if level < self.level:
            return
        if level < Logger.WARNING:
            print(*msg, file=self.out)
        else:
            print(*msg, file=self.err)

    def on(self):
        """
        Redirect all standard print() calls after this method call to the logger.
        However, the notion of levels within standard prints do not exists, so
        we simply redirect the output and do not check whether it is valid.
        """
        sys.stdout = self.out
        sys.stderr = self.err

    def off(self):
        """
        Stop redirecting all standard print() calls after this method call to the logger.
        """
        sys.stdout = self._stdout
        sys.stderr = self._stderr


class Tracker(metaclass=Singleton):
    """
    A Tracker class keeps track of the queries towards the original neural network.

    I.e. whenever the extractor tries a new input on the network and receives an output
    it stores the inputs and outputs.
    """
    def __init__(self):
        """
        Construct a Tracker object.
        """
        self.query_count = 0
        self.query_count_at = {}

        #: All queries we've generated so that we can use them later on. (Format: [(x, f(x))])
        self.saved_queries = []

    def reset(self):
        """
        Reset the tracker to its original state.

        This should be called after completing extraction of a single model,
        to prevent tracking information bleeding over to the next extraction.
        """
        self.query_count = 0
        self.query_count_at = {}
        self.saved_queries = []

    def save_queries(self, queries):
        """
        Save the the input and outputs of the neural network inside the tracker.
        :param queries: A list of the queries passed through the network. (Format: [(x, f(x))])
        """
        self.saved_queries.extend(queries)

    @property
    def nr_of_queries(self):
        """
        The number of queries saved.
        """
        return len(self.saved_queries)
