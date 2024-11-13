#******************************************************************************
#********************************PERFORMANCE TOOLS*****************************
#******************************************************************************

class timeit:
    """
    Class for timing the execution of a program

    CLASS INITIALIZATION:
        
        CALLING SEQUENCE: 
        
            tt = timeit(start=False):
        
        INPUT (OPTIONAL):

            start=False (bool): If start == True, then it start the timer right at the 
            initialization.

    
    """


    def __init__(self,start=False,use_localtime = False):
        if use_localtime:
            from time import localtime as tt
        else :
            from time import time as tt
        self._uselt = use_localtime
        self._tim = tt
        self._times=[]
        if start: self.start()

    def start(self):

        self._t0 = self._tim()
        self._times.append(self._t0)

    def stop(self,message="",verbose=True,hhmmss=False,raw_output = False):

        self._t1 = self._tim()
        self._times.append(self._t1)
        if raw_output : return self._times[-1] - self._times[-2]

        if verbose:
            if self._uselt:
                dt = timediff(self._times[-1], self._times[-2]).total_seconds()
            else:
                dt = self._times[-1] - self._times[-2]

            if hhmmss:
                print(message+"elapsed time: %dh %dm %fs" % (dt//3600,dt%3600//60,dt%3600 % 60 )) 
            else:
                print(message+"elapsed time (sec): %f" % (dt,)) 


    def total_elapsed(self,verbose=True,hhmmss=False,from_1st_start = False):
        
        self._tlast = self._tim()
        _t0 = self._times[0] if from_1st_start else self._t0
        self._dttot = timediff(self._tlast, _t0).total_seconds() if self._uselt else self._tlast - _t0

        if verbose:
            if hhmmss:
                dt=self._dttot
                print("Total elapsed time: %dh %dm %fs" % (dt//3600,dt%3600//60,dt%3600 % 60 )) 
            else:
                print("Total elapsed time (sec): %f" % (self._dttot)) 

    @property
    def tic(self):
        self.start()
    
    @property
    def toc(self):
        self.stop()
    
    @property
    def get_toc(self):
        return self.stop(raw_output = True)

def timediff(a,b):
    """
    Computes the difference between two time.struct_time arrays
    """
    from datetime import datetime
    from time import mktime
    data = datetime.fromtimestamp(mktime(a))
    datb = datetime.fromtimestamp(mktime(b))
    return data-datb
