"""Class used for development purposes in Simulator"""
import math
import os
import string
import random
import pickle
import contextlib
from copy import deepcopy
from itertools import islice
from agent import Agent
from geographiclib import geodesic as geolib_gdsc
from copy import deepcopy
from math import log, exp

@contextlib.contextmanager
def temp_dir(x):
    d = os.getcwd()
    if x is None:
        pass
    # This could raise an exception, but it's probably
    # best to let it propagate and let the caller
    # deal with it, since they requested x
    else: os.chdir(x)

    try:
        yield

    finally:
        # This could also raise an exception, but you *really*
        # aren't equipped to figure out what went wrong if the
        # old working directory can't be restored.
        os.chdir(d)

class DevSim:
    """Mixin to Simulation class for  dev purposes"""

#=================================FOR DEV USE ONLY================================================

    def print_conflicted(self, evo):
        if self.full_history[evo]['conflicted_list']:
            for tup in self.full_history[evo]['conflicted_list']:
                print(([el.id for el in tup[0:2]], tup[2]))

    def print_intersection_status(self):
        print(f'concurrent intersections are {self.current_intersections}')
        for key, val in self.intersection_dict.items():
            print(f'node:{key}')
            for agent in val:
                 print(f'agent: {agent.id}, distance:{agent.intersection[1]}')

    def agent_by_evo(self, evo, agent_id):
        return self.history_save[evo][agent_id]

    def pick_conflict_by_evo(self,evo):
        ll=[]
        for tup in self.full_history[evo]['conflicted_list']:
             ll.append(tup[0:2])
             print(([el.id for el in tup[0:2]], tup[2]))
        return ll

    def rerun_from_evo(self, run_num, evo=0, evolutions_number=None, new_start_data={}):
        """Reruns simulation from a given point, potentially with some parameters altered.

        Args:
            evo(int): evolution number
            new_start_data(dict): in self.full_history structure

        """
        if evolutions_number is None:
            pass
        else:
            self.evolutions_number = evolutions_number

        if new_start_data:
            self.agents_dict = new_start_data['agents']
            self.graph = new_start_data['graph']
        else:
            self.agents_dict = deepcopy(self.runs[run_num].history_save[evo])
            self.history_save = deepcopy(dict(zip(list(self.runs[run_num].history_save)[0:evo],
                                         list(self.runs[run_num].history_save.values())[0:evo])))
            # self.full_history = dict(zip(list(self.full_history)[0:evo], list(self.full_history.values())[
            #                                                                      0:evo]))
        run_number = len(self.runs)

        initial_conditions_export = deepcopy(self.runs[run_num].initial_conditions)
        initial_conditions = dict([tup for tup in initial_conditions_export.items() if tup[1][0] >= evo])

        new_agents_info = None, None, None, initial_conditions

        for self.evolution in range(evo, self.evolutions_number):
            new_agents_info\
                = \
            self.single_evolution_procedure(
                new_agents_info
            )
        self.wrap_the_simulation(initial_conditions_export)
        self.runs[run_number] = Run(run_number, initial_conditions_export, self.iteration_duration,
                                    self.duration_time, self.graph, deepcopy(self.history_save), self.run_stats)
        self.agents_dict = {}

    def _way_to_intersection(self, agent):
        way = list()
        node = agent.edge[1]
        if 'speed' in self.graph.master_dict[node]:
            way.append((node, agent.position, self.graph.master_dict[node]['speed']))
        else:
            way.append((node, agent.position, None))
        while node != agent.intersection[0]:
            node, distance, _ = self.graph.adjacency[node]
            if 'speed' in self.graph.master_dict[node]:
                way.append((node, distance, self.graph.master_dict[node]['speed']))
            else:
                way.append((node, distance, None))
        return way

    def way_to_intersection(self, evolution, agent_id=None):
        agent = self.history_save[evolution][agent_id]
        print(agent_id, self._way_to_intersection(agent))

    def import_initial_conditions(self, path):
        return Run.from_pickle(path).initial_conditions

    def make_f(self, agt):
        def f(k):
            return self.agent_by_evo(k, agt)

        return f

    def agents_fun_analysis(self,run_num):

        for agt in self.runs[run_num].initial_conditions.keys():
            globals()[agt] = self.make_f(agt)
