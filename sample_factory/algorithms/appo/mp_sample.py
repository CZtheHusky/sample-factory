import subprocess
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='dmlab_laser_gen', required=True)
parser.add_argument('--store_path', type=str, default='/nfs/dgx02/raid/dmlab_gen_dataset')
parser.add_argument('--task_start', type=int, default=0)
parser.add_argument('--task_end', type=int, default=108)
parser.add_argument('--gpus', nargs='+', type=int, default=[-1])
parser.add_argument('--collect_target', type=int, default=10000)

def main(args):
    env_name = args.env
    expert_path = args.env
    collect_target = args.collect_target
    store_path = os.path.join(args.store_path, env_name)
    os.makedirs(store_path, exist_ok=True)
    gpu_list = args.gpus
    command = ['python', '-m', 'sample_factory.algorithms.appo.dmlab_sampler', '--no_render', "--algo=APPO", "--dmlab_renderer=software", "--dmlab_one_task_per_worker=True", "--dmlab_level_cache_path=/home/cz/dmlab_cache", "--dmlab_extended_action_set=True", f"--env={env_name}", f"--experiment={expert_path}", f"--store_path={store_path}", f"--traj_num={collect_target}"]
    n_jobs = args.task_end - args.task_start
    processes = []
    for i in range(n_jobs):
        cmd_i = command + [f"--worker_index={i + args.task_start}"]
        print(cmd_i)
        os_env = os.environ.copy()
        gpu_id = gpu_list[i % len(gpu_list)]
        os_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        processes.append(subprocess.Popen(cmd_i, env=os_env))

    for p in processes:
        p.wait()
    
    print(f'{env_name} Done')

if __name__ == '__main__':
    # python -m sample_factory.algorithms.appo.mp_sample --env dmlab_laser_gen --store_path /nfs/dgx02/raid/dmlab_gen_dataset --task_start 0 --task_end 10 --collect_target 10000 --gpus 0 1 2 3 4 7
    args = parser.parse_args()
    main(args)
