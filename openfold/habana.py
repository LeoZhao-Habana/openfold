import os, sys, time

ENABLE_HABANA = False
ENABLE_LAZY_MODE = False
ENABLE_HMP = False

def enable_habana():
    global ENABLE_HABANA
    ENABLE_HABANA = True
    global ENABLE_LAZY_MODE
    ENABLE_LAZY_MODE = True
#    from habana_frameworks.torch.utils.library_loader import load_habana_module
#    load_habana_module()
    import habana_frameworks.torch.core

def is_habana():
    global ENABLE_HABANA
    return ENABLE_HABANA

def enable_lazy_mode():
    global ENABLE_LAZY_MODE
    ENABLE_LAZY_MODE = True
    os.environ["PT_HPU_LAZY_MODE"]="1"
    import habana_frameworks.torch.core

def disable_lazy_mode():
    global ENABLE_LAZY_MODE
    ENABLE_LAZY_MODE = False
    os.environ["PT_HPU_LAZY_MODE"]="2"
    import habana_frameworks.torch.core

def enable_hmp():
    global ENABLE_HMP
    ENABLE_HMP = True

def is_hmp():
    global ENABLE_HMP
    return ENABLE_HMP

def mark_step():
    global ENABLE_LAZY_MODE
    if ENABLE_LAZY_MODE:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()

class habana_timer:
    
    def __init__(self):
        self.st = 0

    def start(self, mod=None):
        self.st = time.time()
        if mod is None:
            mod = sys._getframe().f_back.f_code.co_name
        print("\nStart {} ...".format(mod), end="")

    def end(self, mod=None):
        # return dur in ms
        dur = (time.time() - self.st) * 1000
        if mod is None:
            mod = sys._getframe().f_back.f_code.co_name
        print("  End {} runs {:.2f} ms".format(mod, dur))
        print_peak_memory(mod)

def print_peak_memory(prefix):
    # print(f"{prefix}: {torch.cuda.max_memory_allocated(rank) // 1e6}MB ")
    # print(f"{prefix}: <n/a> : Habana team is working on enabling a mechanism to report memory usage in an upcoming release ")
    global ENABLE_HABANA
    if not ENABLE_HABANA:
        return

    from habana_frameworks.torch.hpu import memory
    used = memory.memory_allocated()
    peak = memory.max_memory_allocated()
    print("memory used: {:.3f} GB, peak: {:.3f} GB".format(used / (1024*1024*1024), peak / (1024*1024*1024)))

    '''
    import habana_frameworks.torch.core as htcore
    msg = prefix + ":"
    htcore.memstat_livealloc(msg)
    '''
