import subprocess
import shlex

if __name__ == '__main__':
    processes_ = []
    subset_dir = '/data/guangyuh/coding_env/ML4PDR/dataset/aag4train/subset_'
    subset_dir_lst = [subset_dir+str(i) for i in range(0,11)]

    for subset in subset_dir_lst:
        processes.append(subprocess.Popen(shlex.split("python main.py --mode 1 -t 18000 -p " + subset + " -c -r on")))
        processes.append(subprocess.Popen(shlex.split("python main.py --mode 1 -t 18000 -p " + subset + " -c -r on -n on -a on -th 0.6 -mn neuropdr_2022-06-09_12:27:41_last")))

    # every process will return a finished infomation
    for p in processes:
        out, err = p.communicate()
        exitcode = p.returncode
        print("exitcode: {}".format(exitcode))
        
    print("end")