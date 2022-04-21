'''
Main function to run PDR (extract the graph as well)
'''
import argparse
import os
from datetime import datetime
from datetime import timedelta
from multiprocessing import Process
from threading import Thread
from time import sleep
import sys
sys.path.append("..")
import model
import pdr
import time
# from env import QL

# When you need to run all folder, setup this
test_file_path = "../dataset/"
test_file_folder_path = "../dataset/aag4train/" #open this if run through a folder
#test_file_path = "../dataset/toy_experiment"
#test_file_path = "../dataset/hwmcc07_amba"
#test_file_path = "../dataset/ILAng_pipeline"
#TODO: 把hwmcc07-10的数据集都跑一次，记录log，pk the SOTA with this framework
#TODO: 尝试一下上次那个用parafoast有争议的case（就是转cnf那个，似乎是ilang simple pipeline 的 stall case？）


#TODO: add a switch to open "generate training set or not"
if __name__ == '__main__':
    #sys.stdout = open('file', 'w') #open this when we need the log
    help_info = "Usage: python main.py <file-name>.aag"
    parser = argparse.ArgumentParser(description="Run tests examples on the PDR algorithm")
    parser.add_argument('fileName', type=str, help='The name of the test to run', default=None, nargs='?')
    parser.add_argument('--mode',type=int,help='choose the mode to run the program, 0 means only run one file, 1 means run through the files in folder',default=0)
    parser.add_argument('-t', type=int, help='the time limitation of one test to run', default=900)
    parser.add_argument('-p', type=str, help='file path of mode 1', default=None)
    parser.add_argument('-c', help='switch to counting time', action='store_true')
    parser.add_argument('-d', type=str, help='switch to do data generation in generalized predecessor or inductive generalization', default='off')
    parser.add_argument('-n', type=str, help='switch to use neural network in inductive generalization or generalized predecessor', default='off')
    parser.add_argument('-a', type=str, help='Use NN-guided IG and append to MIC', default='off')

    # TODO: Add abstract & craig interpolation?
    # TODO: Solve the issue on this case (cannot run in time)
    # TODO: Solve the bug on this case (weird inductive invariant)
    # TODO: Using Dr.Zhang's method to accelerate the speed of solving unsafe case
    # TODO: Solve the bug on hwmcc07_amba/spec1-and-env.aag case (safe -> unsafe)

    '''
    --------------------Run on a folder------------------
    args = parser.parse_args(['--mode', '1', '-t', '5', '-c','-d','off','-n','off','-a','off'])
    '''

    '''
    --------------------Run on a single file------------------
    
    Safe case:
    args = parser.parse_args(['../dataset/aig_benchmark/hwmcc07_tip/nusmv.syncarb5^2.B.aag','-c','-d','off','-n','ig','-a','on'])
    args = parser.parse_args(['../dataset/aig_benchmark/beem_aag/beemadd1b1.aag','-c','-d','off','-n','ig'])
    args = parser.parse_args(['../dataset/aig_benchmark/hwmcc07_tip_aag/vis.4-arbit^1.E.aag','-c','-d','off','-n','ig'])
    args = parser.parse_args(['../dataset/aig_benchmark/hwmcc07_amba/spec1-and-env.aag','-c']) #When you need to run single file, setup this
    args = parser.parse_args(['../dataset/aig_benchmark/hwmcc07_tip/eijk.S208o.S.aag', '-c','-d','off','-n','on','-a','on']) #Found bug here
    
    Unsafe case:
    #SAT, this case can cause memory exploration
    args = parser.parse_args(['../dataset/aig_benchmark/hwmcc07_tip/ken.flash^12.C.aag', '-c','-s','off','-n','ig']) 
    #SAT, time so long, around 20 minutes
    args = parser.parse_args(['../dataset/aig_benchmark/hwmcc07_tip/texas.two_proc^5.E.aag', '-c','-s','off','-n','ig']) 
    
    Get a sorted list of all the files in the folder
    import subprocess
    sp = subprocess.Popen("du -b ../dataset/aag4train/* | sort -n", stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    lst = [line.decode("utf-8").strip('\n').split('\t') for line in sp.stdout.readlines()]
    print(lst)
    
    '''
    
    
    args = parser.parse_args(['../dataset/aag4train/vis.4-arbit^1.E.aag', '-c','-n','on','-a','on']) 
    if (args.fileName is not None) and (args.mode==0):
        file = args.fileName
        m = model.Model()

        state_size = 10  # set up RL
        action_size = 8  # set up RL
        agent = None #QL(state_size, action_size)  # set up RL

        print("============= Running test ===========")

        # Not using RL
        solver = pdr.PDR(*m.parse(file))

        # Switch to turn on/off using neural network to guide generalization (predecessor/inductive generalization)
        if args.n=='off':
            solver.test_IG_NN = 0
            solver.test_GP_NN = 0
        elif args.n=='on':
            solver.test_IG_NN = 1
            solver.test_GP_NN = 1
        elif args.n=='ig':
            solver.test_IG_NN = 1
            solver.test_GP_NN = 0
        elif args.n=='gp':
            solver.test_IG_NN = 0
            solver.test_GP_NN = 1

        # Switch to turn on/off the data generation of generalized predecessor or inductive generalization
        if args.d=='off':
            solver.smt2_gen_GP = 0
            solver.smt2_gen_IG = 0
        elif args.d=='on':
            solver.smt2_gen_GP = 1
            solver.smt2_gen_IG = 1
        elif args.d=='ig':
            solver.smt2_gen_IG = 1
            solver.smt2_gen_GP = 0
        elif args.d=='gp':
            solver.smt2_gen_IG = 0
            solver.smt2_gen_GP = 1

        # On/off the NN-guided ig append to MIC
        if args.a=='off':
            solver.NN_guide_ig_append = 0
        elif args.a=='on':
            solver.NN_guide_ig_append = 1

        startTime = time.process_time()
        solver.run(agent)
        endTime = time.process_time()
        if args.c:
            if solver.NN_guide_ig_time_sum != 0:
                print("TIME CONSUMING IN TOTAL: ",(endTime - startTime) ,"seconds")
                print("TIME CONSUMING: " ,(endTime - startTime - solver.NN_guide_ig_time_sum) , "seconds")  
                print("NN-guided inductive generalization success rate: ",(solver.NN_guide_ig_success/(solver.NN_guide_ig_success + solver.NN_guide_ig_fail))*100,"%")             
            else:
                print("TIME CONSUMING: " ,(endTime - startTime) , "seconds")

        # Using RL
        # for i in range(20):
        #     startTime = time.process_time()
        #     solver = pdr.PDR(*m.parse(file))
        #     solver.run(agent)
        #     endTime = time.process_time()
        #     if args.c:
        #         print("TIME CONSUMING: " + str((endTime - startTime).seconds) + "seconds")

    elif args.mode==1: # 1 means runs through all the folder
        print("================ Test the ./aag directory ========")
        agent = None
        for root, dirs, files in os.walk(args.p): #TODO: 把aig原本的二进制文件也搬进来，这边代码改成仅把.aag加入处理的文件队列里面
            for name in files:
                if name.endswith('.aag'):
                    print("============ Testing " + str(name) + " ==========")
                    m = model.Model()
                    solver = pdr.PDR(*m.parse(os.path.join(root, name)))

                    # Switch to turn on/off using neural network to guide generalization (predecessor/inductive generalization)
                    if args.n=='off':
                        solver.test_IG_NN = 0
                        solver.test_GP_NN = 0
                    elif args.n=='on':
                        solver.test_IG_NN = 1
                        solver.test_GP_NN = 1
                    elif args.n=='ig':
                        solver.test_IG_NN = 1
                        solver.test_GP_NN = 0
                    elif args.n=='gp':
                        solver.test_IG_NN = 0
                        solver.test_GP_NN = 1

                    # Switch to turn on/off the data generation of generalized predecessor or inductive generalization
                    if args.d=='off':
                        solver.smt2_gen_GP = 0
                        solver.smt2_gen_IG = 0
                    elif args.d=='on':
                        solver.smt2_gen_GP = 1
                        solver.smt2_gen_IG = 1
                    elif args.d=='ig':
                        solver.smt2_gen_IG = 1
                        solver.smt2_gen_GP = 0
                    elif args.d=='gp':
                        solver.smt2_gen_IG = 0
                        solver.smt2_gen_GP = 1

                    startTime = time.process_time()
                    timeout = False

                    # t = Thread(target=solver.run)
                    # t.daemon = True
                    # t.start() # start the thread
                    # t.join(timeout=args.t) #FIXME: If timeout, the thread will throw exception and program core dump
                    
                    p = Process(target=solver.run, name="PDR")
                    p.start()
                    # Wait a maximum of 10 seconds for foo
                    # Usage: join([timeout in seconds])
                    p.join(timeout=int(args.t))
                    endTime = time.process_time()
                    # If thread is active
                    if p.is_alive():
                        timeout = True
                        print("PDR run out of the time... let's kill it...")
                        # Terminate foo
                        p.terminate()
                        p.join()

                    # if timeout == True:
                    #     sleep(20)
                    # elif timeout != True:
                    if timeout != True:
                        print("Done in time")
                        #sleep(20)
                        if args.c:
                            print("TIME CONSUMING: ", (endTime - startTime), "seconds")
    else:
        print("Wrong input, please give a vaild input or check the document")

#TODO: Run all the dataset (includes tip..)
