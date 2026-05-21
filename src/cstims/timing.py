from __future__ import annotations

from functools import wraps
from time import perf_counter
from collections import defaultdict
import logging

log = logging.getLogger(__name__)

class FunctionTimer:
    """Tracks timing statistics for decorated functions."""
    
    _enabled = True
    _timings = defaultdict(list)
    
    @classmethod
    def get_stats(cls):
        """Returns dict of function name -> (total_time, call_count, avg_time)."""
        stats = {}
        for func_name, times in cls._timings.items():
            total = sum(times)
            count = len(times)
            avg = total / count if count > 0 else 0.0
            stats[func_name] = {
                'total_time': total,
                'call_count': count,
                'avg_time': avg,
                'min_time': min(times) if times else 0.0,
                'max_time': max(times) if times else 0.0,
            }
        return stats
    
    @classmethod
    def log_summary(cls, level=logging.INFO):
        """Log timing summary statistics."""
        if not cls._timings:
            return
        
        stats = cls.get_stats()
        log.log(level, "\n" + "="*80)
        log.log(level, "Function timing summary:")
        log.log(level, "-"*80)
        log.log(level, f"{'Function':<40} {'Total (s)':<12} {'Calls':<8} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        log.log(level, "-"*80)
        
        for func_name in sorted(stats.keys()):
            s = stats[func_name]
            log.log(level, f"{func_name:<40} {s['total_time']:<12.4f} {s['call_count']:<8} {s['avg_time']:<12.6f} {s['min_time']:<12.6f} {s['max_time']:<12.6f}")
        
        log.log(level, "="*80 + "\n")

def timed(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not FunctionTimer._enabled:
            return func(*args, **kwargs)
        
        start = perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = perf_counter() - start
            FunctionTimer._timings[func.__name__].append(elapsed)
    
    return wrapper
