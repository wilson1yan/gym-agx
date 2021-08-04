import argparse
import os
import multiprocessing as mp
import psutil

def worker(i, n_procs, env_name):
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    assert logical_cores % physical_cores == 0
    assert physical_cores % n_procs == 0
    cpus_per_proc = physical_cores // n_procs
    thread_offset = physical_cores # assumes 2 hyperthreads per core

    cpu_ids = list(range(i * cpus_per_proc, (i + 1) * cpus_per_proc))
    cpu_ids = sum([[cpu_id, cpu_id + thread_offset] for cpu_id in cpu_ids], [])
    cpu_ids = ','.join(map(str, cpu_ids))

    print(f"Running process {i} on GPU {i % 8} with cpus {cpu_ids}")
    cmd = f'DISPLAY=:0.{i % 8} taskset -c {cpu_ids} python worker_taskset.py {i} {n_procs} {env_name}'
    os.system(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="RopeObstacle-v2")
    ap.add_argument("--n_procs", type=int, default=1)
    ap.add_argument("-l", action="store_true", default=False)

    args = vars(ap.parse_args())

    if args["l"]:
        from gym import envs
        all_envs = envs.registry.all()
        env_ids = [env_spec.id for env_spec in all_envs if 'agx' in env_spec.id]
        print("Available AGX environments are: ")
        print(env_ids)
        return

    procs = [mp.Process(target=worker, args=(i, args['n_procs'], args['env']))
             for i in range(args['n_procs'])]
    [p.start() for p in procs]
    [p.join() for p in procs]


if __name__ == "__main__":
    main()
