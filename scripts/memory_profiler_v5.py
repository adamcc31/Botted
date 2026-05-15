"""
scripts/memory_profiler_v5.py
==============================
Memory profiler for V5 state data structures.
Simulates 1000 market cycles to measure growth rate.

Usage:
    python scripts/memory_profiler_v5.py
"""

import sys
import time
import json
import tracemalloc
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def sizeof_object(obj) -> int:
    """Recursively estimate size of a Python object in bytes."""
    import sys
    seen = set()
    def inner(o):
        oid = id(o)
        if oid in seen:
            return 0
        seen.add(oid)
        size = sys.getsizeof(o)
        if isinstance(o, dict):
            size += sum(inner(k) + inner(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            size += sum(inner(i) for i in o)
        return size
    return inner(obj)


def simulate_v5_memory_leak(n_markets: int = 1000):
    """
    Simulate V5 production scenario for n_markets market cycles.
    Tracks actual memory growth of the data structures.
    """
    print(f"=== V5 Memory Profiler: {n_markets} market cycles ===\n")
    
    tracemalloc.start()
    
    # === CURRENT (BROKEN) STATE ===
    # These are exactly the buggy structures from main.py
    session_stats_trades = []          # line 171 - unbounded list
    slingger_daily_stats_pnls = []     # line 176 - unbounded list
    completed_markets = set()          # line 165 - never pruned

    snapshot_sizes = []
    
    for i in range(n_markets):
        market_id = f"market_{i:06d}"
        
        # Simulate trade append - exactly as in _shadow_scalp_monitor_loop
        session_stats_trades.append({
            'result': 'HIT' if i % 2 == 0 else 'MISS',
            'net_pnl': 0.5 if i % 2 == 0 else -1.0,
            'gross_pnl': 0.52 if i % 2 == 0 else -1.0,
            'hold_seconds': 45,
        })
        slingger_daily_stats_pnls.append(0.5 if i % 2 == 0 else -1.0)
        
        # completed_markets grows forever (no pruning in current code)
        completed_markets.add(market_id)
        
        # Every 100 iterations, snapshot sizes
        if (i + 1) % 100 == 0:
            trades_size = sizeof_object(session_stats_trades)
            pnls_size = sizeof_object(slingger_daily_stats_pnls)
            completed_size = sizeof_object(completed_markets)
            total = trades_size + pnls_size + completed_size
            snapshot_sizes.append((i+1, total, trades_size, pnls_size, completed_size))
    
    print("CURRENT (BUGGY) STATE - Memory Growth:")
    print(f"{'Cycle':<8} {'Total(KB)':<12} {'trades(KB)':<12} {'pnls(KB)':<10} {'completed(KB)':<14}")
    print("-" * 60)
    for cycle, total, t, p, c in snapshot_sizes:
        print(f"{cycle:<8} {total/1024:<12.2f} {t/1024:<12.2f} {p/1024:<10.2f} {c/1024:<14.2f}")
    
    # Growth rate calculation
    if len(snapshot_sizes) >= 2:
        first = snapshot_sizes[0]
        last = snapshot_sizes[-1]
        cycles_elapsed = last[0] - first[0]
        bytes_growth = last[1] - first[1]
        bytes_per_market = bytes_growth / cycles_elapsed if cycles_elapsed > 0 else 0
        
        # At 5-min markets, ~12 markets/hour, ~288/day
        markets_per_hour = 12
        bytes_per_hour = bytes_per_market * markets_per_hour
        mb_per_hour = bytes_per_hour / (1024 * 1024)
        
        # Railway limit: 512MB. Assume baseline 200MB.
        available_mb = 512 - 200
        hours_to_oom = available_mb / mb_per_hour if mb_per_hour > 0 else float('inf')
        
        print(f"\nGrowth Rate Analysis:")
        print(f"  Bytes per market cycle:  {bytes_per_market:.1f} bytes")
        print(f"  MB per hour (12 markets): {mb_per_hour:.4f} MB/hr")
        print(f"  Estimated OOM at 512MB limit: {hours_to_oom:.1f} hours")
    
    print("\n")
    
    # === FIXED STATE ===
    print("FIXED STATE - Memory Growth with daily reset + bounded set:")
    
    session_stats_trades_fixed = []
    slingger_daily_stats_pnls_fixed = []
    completed_markets_fixed = set()
    MAX_COMPLETED_MARKETS = 500  # Rolling window
    
    snapshot_sizes_fixed = []
    
    for i in range(n_markets):
        market_id = f"market_{i:06d}"
        
        session_stats_trades_fixed.append({
            'result': 'HIT' if i % 2 == 0 else 'MISS',
            'net_pnl': 0.5 if i % 2 == 0 else -1.0,
            'gross_pnl': 0.52 if i % 2 == 0 else -1.0,
            'hold_seconds': 45,
        })
        slingger_daily_stats_pnls_fixed.append(0.5 if i % 2 == 0 else -1.0)
        completed_markets_fixed.add(market_id)
        
        # FIX 1: Daily reset simulation (every 288 markets = 24h at 12/hr)
        if (i + 1) % 288 == 0:
            session_stats_trades_fixed = []
            slingger_daily_stats_pnls_fixed = []
        
        # FIX 2: Bounded completed_markets (keep only last 500)
        if len(completed_markets_fixed) > MAX_COMPLETED_MARKETS:
            # Remove oldest entries (approximation since set is unordered)
            to_remove = list(completed_markets_fixed)[:len(completed_markets_fixed) - MAX_COMPLETED_MARKETS]
            completed_markets_fixed.difference_update(to_remove)
        
        if (i + 1) % 100 == 0:
            trades_size = sizeof_object(session_stats_trades_fixed)
            pnls_size = sizeof_object(slingger_daily_stats_pnls_fixed)
            completed_size = sizeof_object(completed_markets_fixed)
            total = trades_size + pnls_size + completed_size
            snapshot_sizes_fixed.append((i+1, total, trades_size, pnls_size, completed_size))
    
    print(f"{'Cycle':<8} {'Total(KB)':<12} {'trades(KB)':<12} {'pnls(KB)':<10} {'completed(KB)':<14}")
    print("-" * 60)
    for cycle, total, t, p, c in snapshot_sizes_fixed:
        print(f"{cycle:<8} {total/1024:<12.2f} {t/1024:<12.2f} {p/1024:<10.2f} {c/1024:<14.2f}")
    
    if len(snapshot_sizes_fixed) >= 2:
        first = snapshot_sizes_fixed[0]
        last = snapshot_sizes_fixed[-1]
        cycles_elapsed = last[0] - first[0]
        bytes_growth = last[1] - first[1]
        bytes_per_market = bytes_growth / cycles_elapsed if cycles_elapsed > 0 else 0
        mb_per_hour = (bytes_per_market * 12) / (1024 * 1024)
        hours_to_oom = (512 - 200) / mb_per_hour if mb_per_hour > 0.0001 else 99999
        print(f"\nFixed Growth Rate:")
        print(f"  Bytes per market cycle:   {bytes_per_market:.1f} bytes")
        print(f"  MB per hour (12 markets): {mb_per_hour:.6f} MB/hr")
        print(f"  Estimated OOM at 512MB:   {'NEVER (stable)' if hours_to_oom > 10000 else f'{hours_to_oom:.0f} hours'}")
    
    tracemalloc.stop()
    print("\n=== Profiling Complete ===")


if __name__ == "__main__":
    simulate_v5_memory_leak(n_markets=2000)
