from simulator import PCBangSimulator
import sys

# 경고 메시지 차단
import warnings
warnings.filterwarnings("ignore", message="Ignoring fixed y limits")

# Recursion limit adjustment
sys.setrecursionlimit(10000)

if __name__ == '__main__':
    sim = PCBangSimulator()
    sim.run()
