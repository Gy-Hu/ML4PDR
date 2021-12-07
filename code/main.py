import argparse
import os
from datetime import datetime
from multiprocessing import Process
import sys
sys.path.append("..")
import model
import pdr
# from env import QL

# When you need to run all folder, setup this
test_file_path = "../dataset/"
#test_file_path = "../dataset/toy_experiment"
#test_file_path = "../dataset/hwmcc07_amba"
#test_file_path = "../dataset/ILAng_pipeline"
#TODO: 把hwmcc07-10的数据集都跑一次，记录log，pk the SOTA with this framework
#TODO: 尝试一下上次那个用parafoast有争议的case（就是转cnf那个，似乎是ilang simple pipeline 的 stall case？）

def run_with_limited_time(func, time):
    p = Process(target=func)
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        return False
    return True


if __name__ == '__main__':
    #sys.stdout = open('file', 'w') #open this when we need the log
    help_info = "Usage: python main.py <file-name>.aag"
    parser = argparse.ArgumentParser(description="Run tests examples on the PDR algorithm")
    parser.add_argument('fileName', type=str, help='The name of the test to run', default=None, nargs='?')
    parser.add_argument('-m', type=int, help='the time limitation of one test to run', default=3600)
    parser.add_argument('-c', help='switch to counting time', action='store_true')
    #args = parser.parse_args(['../dataset/hwmcc07_amba/spec1-and-env.aag','-c']) #When you need to run single file, setup this
    #TODO: Add abstract & craig interpolation?
    args = parser.parse_args(['../dataset/hwmcc07_tip/ken.flash^12.C.aag', '-c']) #SAT, this case can cause memory exploration
    #TODO: Using Dr.Zhang's method to accelerate the speed of solving unsafe case
    #args = parser.parse_args(['../dataset/hwmcc07_tip/texas.two_proc^5.E.aag', '-c']) #SAT
    #args = parser.parse_args(['../dataset/hwmcc07_tip/nusmv.syncarb5^2.B.aag','-c'])
    #args = parser.parse_args(['../dataset/hwmcc07_tip/eijk.S208o.S.aag', '-c'])
    #args = parser.parse_args(['../dataset/ILAng_pipeline/simple_pipe_verify_stall_ADD.aag', '-c'])
    #args = parser.parse_args(['../dataset/toy_experiment/counter_unsat.aag', '-c'])
    #args = parser.parse_args(['../dataset/toy_experiment/play.aag', '-c'])
    #args = parser.parse_args(['-c']) #Run through a folder
    if args.fileName is not None:
        file = args.fileName
        m = model.Model()

        state_size = 10  # set up RL
        action_size = 8  # set up RL
        agent = None #QL(state_size, action_size)  # set up RL

        print("============= Running test ===========")

        # Not using RL
        solver = pdr.PDR(*m.parse(file))
        startTime = datetime.now()
        solver.run(agent)
        endTime = datetime.now()
        if args.c:
            print("TIME CONSUMING: " + str((endTime - startTime).seconds) + "seconds")

        # Using RL
        # for i in range(20):
        #     startTime = datetime.now()
        #     solver = pdr.PDR(*m.parse(file))
        #     solver.run(agent)
        #     endTime = datetime.now()
        #     if args.c:
        #         print("TIME CONSUMING: " + str((endTime - startTime).seconds) + "seconds")

    else:
        print("================ Test the ./aag directory ========")
        for root, dirs, files in os.walk(test_file_path): #TODO: 把aig原本的二进制文件也搬进来，这边代码改成仅把.aag加入处理的文件队列里面
            for name in files:
                if name.endswith('.aag'):
                    print("============ Testing " + str(name) + " ==========")
                    m = model.Model()
                    solver = pdr.PDR(*m.parse(os.path.join(root, name)))
                    startTime = datetime.now()
                    if not run_with_limited_time(solver.run, args.m):
                        print("Time Out")
                    else:
                        endTime = datetime.now()
                        print("Done in time")
                        if args.c:
                            print("TIME CONSUMING: " + str((endTime - startTime).seconds) + "seconds")

#TODO: Run all the dataset (includes tip..)
