import os
from environment import BaseGraph
from sim import Simulation
from vis import vis
import pickle
import argparse

def list_of_ints(arg):
    l = arg.split(',')
    return [int(el) for el in l]

parser = argparse.ArgumentParser(
                    prog='STARsim_starter',
                    description='Program to initiate simulations within STARsim')
parser.add_argument('-T', '--time', type=str, required=True, metavar='01:30:00', help='Simulation Duration Time as a '
                                                                                     'string in HH:MM:SS '
                                                                  'formatt')
parser.add_argument('-D', '--data', type=str, required=True, help='Path to folder with data')
parser.add_argument('-N', '--runs_number', type=int, required=True, help='Number of simulations to run')
parser.add_argument('-V', '--visualise', type=list_of_ints, metavar='1,4,5',  help='list of runs to visualise (count starts '
                                                                          'with '
                                                                       '1)')
parser.add_argument('-S', '--stats', type=str, help='Number of simulations to run')
group = parser.add_mutually_exclusive_group()
group.add_argument('-i', '--initial', help='Name of the file with initial conditions')
group.add_argument('-n', '--agents_number', type=int, help='Number of aggents to participate in the '
                                                                         'simulations, if provided the initial '
                                                                         'sequence will have '
                                                                         'agents\' starting times drawn randomly')

args = parser.parse_args()

def starter(args):
    global graph
    graph = BaseGraph.from_csv(os.path.expanduser(args.data), elevation=345)
    graph.prepare_for_simulation()

    sim = Simulation(use_time=False, environment=graph, iteration_duration=1)
    if args.initial:
        sim.starter(
            read_starting_times=os.path.join(args.data,args.initial),
            duration_time=args.time, perturbed=True, runs_number=args.runs_number,
            # agents_number=20,
            shortcuts_allowed=True
        )
    elif args.agents_number:
        sim.starter(
            # read_starting_times='/home/kk/Desktop/Studia/PracMag/data2/bez_poch.csv',
            duration_time=args.time, perturbed=True, runs_number=rgs.runs_number,
            agents_number=20,
            shortcuts_allowed=True
        )
    else:
        raise ValueError('Neither initial conditions, nor number of agents where provided')

    if args.visualise is not None:
        for i in args.visualise:
            vis(sim,i-1)

    if args.stats:
        export_stats={}
        for key, run in sim.runs.items():
            export_stats[key] = run.stats
        with open(args.stats + '.pickle', 'wb') as f:
            pickle.dump(export_stats, f)


if __name__ == '__main__':
    starter(args=args)
