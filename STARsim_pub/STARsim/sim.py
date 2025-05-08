"""" Simulator class"""

import pickle
import csv
import random
from random import normalvariate as nv
import string
from copy import deepcopy
from re import match
import contextlib

import pandas as pd
from geographiclib import geodesic as geolib_gdsc

from DevSim import DevSim#, temp_dir
from SpaceBasedParadigm import SpaceBasedParadigm, grad_generator
from TimeBasedParadigm import TimeBasedParadigm
from agent import Agent

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

def timestring_to_seconds(timestr):
    """Converts time string of a following form 'HH:mm:ss' to duration time in seconds

    Args:
        timestr(str): time as a string (hh:mm:ss)

    Returns:
        time in seconds (int)
    """
    if timestr is None:
        return None
    else:
        ftr = [3600, 60, 1]
    return sum([a * b for a, b in zip(ftr, map(int, timestr.split(":")))])


"""for developmnet purposes we mixin DevSim as we want this methods to be separated 
to delete them afterwards"""


class Simulation(DevSim, SpaceBasedParadigm, TimeBasedParadigm):
    """Class encapsulating methods and attributes needed for a single simulation run.

    Constructors:
        __init__()

    Methods:
        run()
        single_evolution_procedure()
        make_a_move()
    """

    def __new__(cls, use_time=True, *args, **kwargs):

        cls_properties = {
            name: getattr(cls, name) for name in dir(cls) if not match("__.*__", name)
        }
        cls_properties['__init__'] = getattr(cls, '__init__')

        if use_time:
            return type(cls.__name__, (DevSim, TimeBasedParadigm, SpaceBasedParadigm), cls_properties)(*args,
                                                                                                       **kwargs)
        else:
            return type(cls.__name__, (DevSim, SpaceBasedParadigm, TimeBasedParadigm), cls_properties)(*args,
                                                                                                       **kwargs)

    def __init__(
            self,
            environment,
            iteration_duration=5,
            duration_time=None,
            absolute_minima=(1000, 5),
            on_graph=True,
    ):
        """Constructor for Simulation class

        Args:
            environment(object): Instance of a BaseGraph class. Environment for the simulation.
            iteration_duration(int): Amount of time it takes for one Simulation evolution to take place
            duration_time(str): Time of simulation duration (format: (hh:mm:ss))
            absolute_minima(tuple): (vertical[ft], horizontal[nm]) Actual minima for the airspace not to be exceeded in
                                    the simulation
            on_graph(bool): When True new agents appear at the starting point of the procedure assigned to them
        """
        self._export = {}
        self.runs = {}
        self.agents_dict = {}
        self.iteration_duration = iteration_duration / 3600
        self._history = []
        self._procedure_agent_dict = dict(
            [(key, []) for key in environment.procedures_dict.keys()]
        )
        self.graph = environment
        self.evolution = 0
        self.duration_time = timestring_to_seconds(duration_time)
        self.history_save = {}
        self.full_history = {}
        self.conflicted_list = []
        self.incidents = []
        self.evolutions_number = None
        self.agents_starting_times = None
        self.agents_off_times = {}
        self.on_graph = True
        self.sorted_agents_tuples = None
        self.absolute_minima = absolute_minima
        self.queuing_only = True
        #TODO sparametryzować poniższe
        self.shortcuts_dict = {
                               # 'NEGEN': 'WA534',
                               'OTMUL': 'WA533',
                               'ADINI': 'WA533',
                               'SISLU': 'WA534',
                                # 'FOLFA': 'NEGEN',
                                # 'WA884': 'SISLU'
                               # 'LEMBU': 'WA536'
                               }
        self.shortcut_pause = 5  # Pause so that two shortcuts cannot be gratned in consecutive evolutions
        self.intersection_dict = dict()
        self.taken_shortcut = set()
        self.minima = (1000, 7)  # can be incorporated in params
        self.params = {'intersection_conflict_limit': self.minima[1], 'minimal_separation': absolute_minima[1]}
        self.current_intersections = self.graph.intersections_set
        self.shortcuts_allowed = None
        self.agents_in_holding_register = {}
        self.shortcuts_register = dict(zip(list(self.shortcuts_dict.items()), [0 for _ in self.shortcuts_dict]))
        self.run_stats = {}
        self.inc_count_no_hold = 0

    def starter(
            self,
            read_starting_times=None,
            initial_conditions=None,
            duration_time=None,
            perturbed = False,
            runs_number=1,
            agents_starting_times=None,
            agents_number=None,
            absolute_minima=(1000, 5),
            on_graph=True,
            shortcuts_allowed=True
    ):
        """

        Args:
            read_starting_times(path): path to the csv file with agents_id's and their respective starting times and
                                        procedures.
            initial_conditions(dict): Pairs of agents_id: (appearance evolution's number, Procedure of choice))
            duration_time(int):
            runs_number(int):
            agents_starting_times(list): Agent's starting times (no procedures, no ids)
            agents_number: Number of agents to generate artifitial initial conditions for the run (when neither
                            initial_conditions nor agents_starting_times given)

        """
        # overwritting attributes set in __init__() if new parameters were given
        self.overwrite_attributes_for_run(on_graph, duration_time, absolute_minima)
        if read_starting_times:
            with open(read_starting_times) as f:
                initial_conditions = dict()
                for line in csv.reader(f):
                    initial_conditions[line[0]] = (
                        int(timestring_to_seconds(line[1]) // (self.iteration_duration * 3600)),
                        line[2]
                    )

        # setting params with random values if not provided
        elif initial_conditions:
            pass
        else:
            initial_conditions = self.set_missing_parameters(
                agents_starting_times,
                agents_number,
            )[0]
        if perturbed:

            prtb_initial_conditions_list = [dict([(key, (int(item[0]
                                                            + max(0, nv(0,(5*60)/(self.iteration_duration*3600)))),
                                                        item[1]))
                    for key,item in initial_conditions.items()])
                    for _ in range(runs_number)
                ]
            for i, intl_cndt in enumerate(prtb_initial_conditions_list):
                self.run(intl_cndt, i, shortcuts_allowed=shortcuts_allowed)

    def set_missing_parameters(self, agents_starting_times, agents_number):
        """sets simulation attributes if not there, if either of the arguments is None it will be set automatically
        on basis of another
        Args:
            agents_starting_times(dict or list): Information on time of agents' appearances in the simulation
            agents_number(int): number of agents in the simulation
        """
        if type(agents_starting_times) == list():
            agents_number = len(agents_starting_times)
            initial_conditions = dict(
                zip(
                    [self.agents_random_id() for _ in range(agents_number)],
                    zip(agents_starting_times,
                        [random.choice(list(self.graph.procedures_dict.keys())) for _ in range(agents_number)],
                        )
                )
            )

        elif type(agents_starting_times) == dict:
            initial_conditions = dict(
                zip(
                    list(agents_starting_times.keys()),
                    zip(list(agents_starting_times.values()),
                        [random.choice(list(self.graph.procedures_dict.keys())) for _ in range(agents_number)],
                        )
                )
            )

        # if no agents_starting dict provided, starting times will be drawn for every agent:
        elif agents_starting_times == None:

            # if number of agents not provided, it will be drawn (upper limit is arbitrary)
            if agents_number is None:
                agents_number = int(random.random() * (self.duration_time / 60))

            initial_conditions = dict(
                zip(
                    [self.agents_random_id() for _ in range(agents_number)],
                    zip(random.sample(range(0, self.evolutions_number), agents_number),
                        [random.choice(list(self.graph.procedures_dict.keys())) for _ in range(agents_number)],
                        )
                )
            )

        return initial_conditions, agents_number

    def for_run_param_reset(self):
        self.full_history = {}
        self.shortcuts_register = dict(zip(list(self.shortcuts_dict.items()), [0 for _ in self.shortcuts_dict]))
        self.current_intersections = self.graph.intersections_set
        self.inc_count_no_hold = 0
        self.agents_dict = {}

    def run(
            self,
            initial_conditions,
            run_number=None,
            agents_starting_times=None,
            agents_number=None,
            try_to_keep_minima=(1500, 7),
            shortcuts_allowed=True
    ):
        """Method responsible for running the simulation
        Args:
            initial_conditions(dict): dictionary of pairs (agent's_id, appearance_evolution)
            on_graph(bool): Information on where should the agents in the simulation appear if True, agents will start
                from the very first node of the procedure, if False they will start off the graph.
            try_to_keep_minima(tuple): (vertical[ft], horizontal[nm]) minima we would like to keep in the simulation
                which can differ from official minima for the airspace
            random_times(bool): If agents_starting_times not provided, function will generate it, when doing it it
                will look at random times to know if it should generate times of appearance at random.
            agents_number(int): number of agents in the simulation
        """
        self.for_run_param_reset()
        # Storing original starting_times as an attribute and sorting original dict to use it as a queue
        # of (key, value) pairs in value ascending order
        if initial_conditions:
            initial_conditions = dict(
                sorted(initial_conditions.items(), key=lambda x: x[1], reverse=True)
            )
        elif agents_number:

            initial_conditions = self.set_missing_parameters(
                None,
                agents_number,
            )[0]
        else:
            initial_conditions = self.set_missing_parameters(
                None,
                None,
            )[0]

        if run_number:
            pass
        else:
            run_number = len(self.runs)

        if shortcuts_allowed:
            self.shortcuts_allowed = True
        else:
            self.shortcuts_allowed = False

        initial_conditions_export = deepcopy(initial_conditions)
        new_agents_info = None, None, None, initial_conditions

        for self.evolution in range(self.evolutions_number):
            new_agents_info = \
                self.single_evolution_procedure(
                    new_agents_info
                )

        self.wrap_the_simulation(initial_conditions_export)
        self.runs[run_number] = Run(run_number, initial_conditions_export, self.iteration_duration,
                                    self.duration_time, self.graph, deepcopy(self.history_save), self.run_stats)
    def single_evolution_procedure(self, new_agents_info):
        """Running all methods that should be run for every simulation's evolution
        Args:
            new_agents_info(tuple): information on new agents to be added (id, evolution of apperance, procedure,

        """

        new_agents_info = self.add_agent(
            *new_agents_info
        )
        self.agents_param_and_info_update()
        self.make_a_move()
        self.history_save[self.evolution] = deepcopy(self.agents_dict)
        self.full_history[self.evolution] = deepcopy(dict((('graph', self.graph), ('agents', self.agents_dict),
                                                           ('conflicted_list', self.conflicted_list),
                                                           ('holding_register', self.agents_in_holding_register),
                                                           ('incidents', self.incidents))))
        # self.adjust_speed()

        return new_agents_info

    def make_a_move(self):
        """Moves all the participants of the simulation

        Args:
            absolute_minima(tuple): (vertical, horizontal) as described in py:meth:`.Simulator.run`
        """
        self.param_reset()
        if self.agents_dict:
            self.agents_FAP_distance_sort()
            for agent_tuple in list(self.agents_dict.items()):
                self.agent_update(agent_tuple[1])

    def FAP_distance(self, agent: Agent=None, node=None):
        """Measures agents distance to FAP"""
        if node is not None:
            return self.graph.master_dict[node]["FAP_distance"]
        elif agent is not None:
            node = agent.edge[1]
            return agent.position + self.graph.master_dict[node]["FAP_distance"]

    def limits_check(self, **kwargs):
        """checking the speed_limits on agents way, starting next node"""
        limit = None
        if 'agent' in kwargs:
            agent = kwargs['agent']
            next_node = agent.edge[1]
            distance = agent.position
        elif 'node' in kwargs:
            next_node = kwargs['node']
            distance = 0
        while limit is None:
            if "speed" in self.graph.master_dict[next_node] and distance > 0:
                limit = self.graph.master_dict[next_node]["speed"]
            else:
                distance += self.graph.adjacency[next_node][1]
                next_node = self.graph.adjacency[next_node][0]
        return limit, next_node, distance

    def update_altitude_limits(self, agent: Agent):
        """search for the next altitude limit on the Graph"""
        limit = None
        distance = agent.position
        next_node = agent.edge[1]

        while limit is None:

            if "agl" in self.graph.master_dict[next_node]:
                limit = self.graph.master_dict[next_node]["agl"]
            else:
                distance += self.graph.adjacency[next_node][1]
                next_node = self.graph.adjacency[next_node][0]

        agent._next_alt_limit, agent._next_node_alt_limit, agent._distance_to_alt_limit = (
            limit,
            next_node,
            distance,
        )
        return next_node, limit, distance

    def agents_param_and_info_update(self):
        """Runs procedures responsible for refreshing parameters and information on agents"""
        for agt in self.agents_dict.values():
            agt.hold = False
        self.agent_single_evo_param_reset()
        self.neighbours_info_update()
        self.adjust_speed()


    def agent_single_evo_param_reset(self):
        """Paramer reset carried out before every evolution"""
        for agent in self.agents_dict.values():
            agent.adjusted = 0
            agent.adjusted_vox = None
            agent.speed_fix = False
            agent.hold = False

    def param_reset(self):
        self.incidents=[]

    def add_agent(self, new_agent, start_ev, procedure, agents_starting_times):
        """Picks first agent candidate from the appearance evolution sorted queue with @pick_new_agent (from
        agents_starting_times: dict) and adds it to the simulation if the evolution of appearance provided
        for the candidate is equal to current evolution's number

        Args:
            new_agent(str): Agents to add name
            start_ev(int): Evolution's number for new_agent appearance
            agents_starting_times(dict): All future agents ascending sorted appearances schedule used as a queue

        Returns:
            new_agent, start_ev: (if not added in the current evolution >> needs to wait for it's time)
            None, None: if agents was added to the simulation
        """
        if agents_starting_times:
            if new_agent is None:
                new_agent, start_ev, procedure = self.pick_new_agents(
                    agents_starting_times, evolution=self.evolution
                )
        if start_ev == self.evolution:
            self.new_agent(on_graph=self.on_graph, id=new_agent, procedure=procedure)
            # new_agent, start_ev, procedure = None, None, None
            if agents_starting_times:
                new_agent, start_ev, procedure = self.pick_new_agents(
                    agents_starting_times, evolution=self.evolution
                )

        return new_agent, start_ev, procedure, agents_starting_times

    def pick_new_agents(self, agents_starting_times, evolution):
        """Function enhancing @add_agent function. Picks agents and their appearance times from agents_starting_times
        queue. Recursive formula ensures the problem of two agents appearing during the same evolution

        Args:
          agents_starting_times
          evolution

        Returns:
            new_agent, start_ev: if not added in the current evolution with the recursion>> needs to wait for it's
            time
            None, None: if agents was added to the simulation
        """
        new_agent, (start_ev, procedure) = agents_starting_times.popitem()

        if start_ev == evolution:
            self.new_agent(id=new_agent, procedure=procedure, on_graph=self.on_graph)
            if agents_starting_times:
                new_agent, start_ev, procedure = self.pick_new_agents(
                    agents_starting_times, evolution
                )
            else:
                return None, None, None
        return new_agent, start_ev, procedure

    def locate_agent(self, agent: object):
        """"Finds agent's coordinates on the basis of his "on graph" / simulation environment position

        Args:
            agent(object): Agent class' instance

        Returns:
            Latitude, Longitude: Agents coordinates on the geodesic
        """
        prev_node, next_node = agent.edge
        azimuth = (geolib_gdsc.Geodesic.WGS84.Inverse(*self.graph.master_dict[prev_node]['coordinates'],
                                                      *self.graph.master_dict[next_node]['coordinates'])['azi2']
                   + 180) % 360
        location_dict = geolib_gdsc.Geodesic.WGS84.Direct(
            *self.graph.master_dict[next_node]["coordinates"],
            azimuth,
            agent.position * 1852,
        )
        return location_dict["lat2"], location_dict["lon2"]

    def agents_random_id(self):
        """Assigns agents id value randomly according to the pattern"""
        id = "".join(random.choice(string.ascii_uppercase) for _ in range(2)) + "".join(
            str(random.randint(0, 9)) for _ in range(4)
        )
        if id in self.agents_dict:
            id = self.agents_random_id()
        return id

    def new_agent(
            self,
            id,
            altitude=20000,
            speed=280,
            start_node=None,
            procedure=None,
            on_graph=True,
            position=None,
            coordinates=None,
    ):
        """Creates new agent entity to be added to the Simulation (to Simulation.agents_dict)"""

        if on_graph:
            if procedure is None:
                procedure = random.choice(list(self.graph.procedures_dict.keys()))
                self._export[id] = [self.evolution, procedure]
            if start_node is None:
                start_node = self.graph.procedures_dict[procedure].loc[
                    1, "TERMINATOR_WAYPOINT"
                ]
            destination_node = self.graph.adjacency[start_node]

        elif coordinates:
            if procedure is None:
                procedure = self.find_closest_procedure(coordinates)
            else:
                pass
            destination_node = self.graph.procedures_dict[procedure].loc[
                1, "TERMINATOR_WAYPOINT"
            ]
            start_node = id
            self.graph.create_temporary_edge(
                start_node, coordinates, altitude, destination_node
            )

        else:
            coordinates, procedure = self.agent_random_appearance_radius()
            destination_node = self.graph.procedures_dict[procedure].loc[
                1, "TERMINATOR_WAYPOINT"
            ]
            start_node = id
            self.graph.create_temporary_edge(
                start_node, coordinates, altitude, destination_node
            )

        self.agents_dict[id] = Agent(id)
        self.agents_dict[id].position = self.graph.adjacency[start_node][1]
        self.agents_dict[id].edge = (start_node, self.graph.adjacency[start_node][0])
        self.agents_dict[id].speed = speed
        self.agents_dict[id].altitude = altitude
        self.agents_dict[id].procedure = procedure
        self._procedure_agent_dict[procedure].append(self.agents_dict[id])
        self.prepare_for_simulation(self.agents_dict[id])


    def agent_random_appearance_radius(self, rad=90, off=None):
        """If agent is to appear off the graph in certain radius off the airport, this method will assign it its
        coordinates and provide with the closest procedure to join"""
        if off:
            rad = rad + 2 * (random.random() - 1 / 2) * off
        if self.graph.airport_coordinates is not None:
            measure_from = self.graph.airport_coordinates
        else:
            measure_from = self.graph.master_dict[self.graph.final_node]["coordinates"]
        azimuth = random.random() * 360
        location_dict = geolib_gdsc.Geodesic.WGS84.Direct(
            *measure_from, azimuth, rad * 1852
        )
        coords = (location_dict["lat2"], location_dict["lon2"])

        return coords, self.find_closest_procedure(coords)


    def find_closest_procedure(self, coords):
        """Finds which prodedure's starting point is the closest to the given coordinates

        Args:
            coords(tuple): geodesic coordinates (latitude, longitude)

        Returns:
            closest procedure's name

        """
        closest_procedure = None
        for procedure in self.graph.procedures_dict:
            pcdr_start = self.graph.procedures_dict[procedure].loc[
                1, "TERMINATOR_WAYPOINT"
            ]
            temp_dist = geolib_gdsc.Geodesic.WGS84.Inverse(
                *coords, *self.graph.master_dict[pcdr_start]["coordinates"]
            )["s12"]
            if closest_procedure is None or temp_dist < closest_procedure[1]:
                closest_procedure = procedure, temp_dist
            else:
                pass

        return closest_procedure[0]

    def geodesic_distance(self, coords1, coords2):
        return geolib_gdsc.Geodesic.WGS84.Inverse(*coords1, *coords2)["s12"] / 1852

    def agents_distance(self, agent1, agent2):
        """Finds distance between to given agents

        Args:
            agent1: Agent class' instance
            agent2: Agent class' instance

        Returns:
            geodesic distance between to agents in [nm]

        """
        if not agent1.coordinates:
            agent1.coordinates = self.locate_agent(agent1)

        if not agent2.coordinates:
            agent2.coordinates = self.locate_agent(agent2)

        # the final result is being devided by 1852, because our base metric are nautical miles and geo_lib works on
        # meters
        return self.geodesic_distance(agent1.coordinates, agent2.coordinates)

    def agents_altitude_compraison(self, minimum=1000):
        """Returns information on agents in potential conflict, due to no altitude separtaion.
        The output is a dict where keys are agent.id's and values are lists of agent.id's

        Args:
            minimum(int): minimum vertical separation to be maintained in [ft]

        Returns:
            danger_dict(dict): dict with pairs of agents in potential conflict (without repetitions)
        """
        danger_dict = dict()
        altitudes_sort = dict(
            sorted(self.agents_dict.items(), key=lambda x: x[1].altitude, reverse=True)
        )
        temp = list(altitudes_sort.items())
        for key, value in altitudes_sort.items():
            danger = []
            idxx = 1
            bound = len(temp)
            while idxx < bound:
                if value.altitude - temp[idxx][1].altitude < minimum:
                    danger.append(temp[idxx][0])
                    idxx += 1
                    if idxx == bound and danger:
                        danger_dict[key] = danger
                        temp.pop(0)
                        break
                elif danger:
                    danger_dict[key] = danger
                    temp.pop(0)
                    break
                else:
                    temp.pop(0)
                    break

        return danger_dict

    def agents_FAP_distance_sort(self, as_fun=False, reverse=False):
        """
        Sorts agents according to the distance to the last node on the graph
        Args:
            as_fun: if True serves as a function returning dict value, else will make changes to its mother Object
                    in agents_sorted_tuples attribute
            reverse: if True will sort in reverse order

        Returns:
            None
            or
            if as_fun set to True a list of tuples (agent_id, agent) sorted according to the distance to last node
            in the graph
        """
        if as_fun:
            return list(
                sorted(
                    self.agents_dict.items(), key=lambda x: self.FAP_distance(x[1]), reverse=reverse,
                )
            )
        else:
            self.sorted_agents_tuples = list(
                sorted(
                    self.agents_dict.items(), key=lambda x: self.FAP_distance(x[1]), reverse=True,
                )
            )

    def overwrite_attributes_for_run(self, on_graph, duration_time, absolute_minima):
        """overwritting attributes for run"""
        if absolute_minima:
            self.absolute_minima = absolute_minima
        if on_graph:
            self.on_graph = True
        # Setting / overwriting parameters of simulation dependent on how the method was called
        if self.duration_time is None and duration_time is None:
            raise ValueError("Simulation duration time was not provided")
        elif self.duration_time is None:
            self.duration_time = timestring_to_seconds(duration_time)
        self.evolutions_number = int(
            self.duration_time // (self.iteration_duration * 3600)
        )

    def agent_next_intersection(self, agent=None, node=None, dist=0):
        """
        Finds next intersection, takes either agent or node and dist as argument
        Args:
            agent:
            node: node to start
            dist: Distance to the node to start (eg when on the shortcut)

        Returns:
            Intersection node, distance to the intersection pair

        """
        if agent:
            next_node = agent.edge[1]
            distance = agent.position
        if node:
            next_node = node
            distance = dist
        while next_node not in self.current_intersections:
            next_node, node_from_node_distance, _ = self.graph.adjacency[next_node]
            distance += node_from_node_distance
        if agent:
            agent.intersection = next_node, distance
        return next_node, distance

    def neighbours_info_update(self):
        """Method updating information about each agent's neighbouring agents with distances to them, for all agents
        participating in the simulation at the given time (evolution). It loops over all the intersections with
        agents, whence it covers all the agents in a single run.
        """
        self.conflicted_list = list()
        self.intersection_dict, beg_dict = self.by_intersection()
        for end, agents_sorted_list in self.intersection_dict.items():

            for idx in range(len(agents_sorted_list) - 1):
                agent: Agent = agents_sorted_list[idx]
                potential_conflict_candidate: Agent = agents_sorted_list[idx + 1]
                conflicted_agents_distance = self.FAP_distance(potential_conflict_candidate) - self.FAP_distance(
                    agent)

                if conflicted_agents_distance < self.params['intersection_conflict_limit']:
                    self.conflicted_list.append((agent, potential_conflict_candidate, conflicted_agents_distance,
                                                 False))

                # The below might not be valuable but stays for now
                agent.potential_conflict_behind = (
                    potential_conflict_candidate, conflicted_agents_distance)
                potential_conflict_candidate.potential_conflict_ahead = (
                    agent, conflicted_agents_distance)

            selection = list(reversed(agents_sorted_list))
            candidate = None
            agent = None
            front = []
            left_overs = []
            lor = None
            while selection or lor:
                if not selection:
                    selection = lor
                if not (agent or candidate):
                    if len(selection) == 1:
                        agent = selection.pop()
                        front.append(agent)
                        continue
                    else:
                        agent, candidate = selection.pop(), selection.pop()
                        front.append(agent)
                elif candidate and not agent:
                    agent, candidate = candidate, selection.pop()
                elif agent and not candidate:
                    candidate = selection.pop()
                increment_node = candidate.edge[0]
                while increment_node != candidate.intersection[0]:
                    if agent.edge[0] == increment_node:
                        # print(f'agents{agent.id, candidate.id}')
                        # print(f'Left overs{[agt.id for agt in left_overs]}')
                        agent.consecutive_agent = (
                            candidate, candidate.intersection[1] -
                            agent.intersection[1])
                        candidate.ahead_agent = (
                            agent, candidate.intersection[1] -
                            agent.intersection[1])
                        agent = None
                        break
                    else:
                        increment_node = self.graph.adjacency[increment_node][0]
                else:
                    left_overs.append(candidate)
                    candidate = None
                    if selection:
                        continue
                if not selection and left_overs:
                    candidate = None
                    agent = None
                    lor = list(reversed(left_overs))
                    left_overs = []

            # można zmienić agt na front[0] bo zawsze powinni mieć tego samego
            for num, agt in enumerate(front):
                agt.ahead_agent = self.ahead_agent_search(agt, beg_dict)
                if num == 0 and agt.ahead_agent[0] is not None and agt.ahead_agent[1] < self.params[
                    'intersection_conflict_limit']:
                    self.conflicted_list.insert(0,(agt.ahead_agent[0], agt, agt.ahead_agent[1], True))



    def by_intersection(self):
        """Groups agents by the next intersection on their way. The other functionality is
        creation of beg_dict which is a dictionary holding information about the first agent after the intersection

        Returns:
            tuple of dictionaries (intersection_dict, beg_dict)

                The first dict's keys are intersection nodes names and its values are unsorted lists of Agent class objects,
                representing agents for which a given node is the next intersection on their way.

                The second dict's keys are also intersection nodes names and the values are Agent class objects, representing
                agents which were the last to cross the given intersection.
        """
        intersection_dict = dict()
        beg_dict = dict()
        for _, agent in self.agents_dict.items():
            node = self.agent_next_intersection(agent)[0]
            beg = agent.edge[0]
            if node in intersection_dict.keys():
                intersection_dict[node].append(agent)
            else:
                intersection_dict[node] = [agent]
            if beg in beg_dict.keys():
                beg_dict[beg].append(agent)
            else:
                beg_dict[beg] = [agent]

        for node, agents in intersection_dict.items():
            intersection_dict[node] = sorted(agents, key=lambda x: x.intersection[1])
        for beg, agents in beg_dict.items():
            beg_dict[beg] = sorted(agents, key=lambda x: x.intersection[1])
        return intersection_dict, beg_dict

    def by_node_dict(self):
        """
        Returns:
            Two dicts, where values are agents grouped in lists according to the edge they are one.
            From the edge (beg,end)
            beg_dict has beg as its keys and end_dict will have end as a key
        """
        end_dict = dict()
        beg_dict = dict()
        for _, agent in self.agents_dict.items():
            beg, end = agent.edge
            if end in end_dict.keys():
                end_dict[end].append(agent)
            else:
                end_dict[end] = [agent]
            if beg in beg_dict.keys():
                beg_dict[beg].append(agent)
            else:
                beg_dict[beg] = [agent]
        for end, agents in end_dict.items():
            end_dict[end] = sorted(agents, key=lambda x: x.position)

        for beg, agents in beg_dict.items():
            beg_dict[beg] = sorted(agents, key=lambda x: x.position, reverse=True)

        return end_dict, beg_dict

    def ahead_agent_search(self, agent, beg_dict):
        node = agent.edge[0]
        if len(beg_dict[node]) > 1:
            beg_list = sorted([agt for agt in beg_dict[node]
                               if agt.edge == agent.edge], key=lambda x: x.position)
            beg_list = [agt for agt in beg_list if not agt.on_shortcut]
            if len(beg_list) > 1:
                agent_list_pos = beg_list.index(agent)
                if agent_list_pos:
                    candidate = beg_list[agent_list_pos - 1]
                    ahead = candidate, self.FAP_distance(agent) - self.FAP_distance(candidate)
                    return ahead
        if agent.on_shortcut:
            node = self.shortcuts_dict[node]
        else:
            node = self.graph.adjacency[node][0]
        ahead = None
        while ahead is None:
            while node not in beg_dict:
                if node == 'EOG':
                    ahead = None, None
                    break
                else:
                    node = self.graph.adjacency[node][0]
                    continue
                break
            # shortcut guys in consideration as well
            else:
                beg = [el for el in beg_dict[node] if el.in_holding]
                if beg:
                    beg = sorted(beg, key=lambda x: x.hold_timer, reverse=True)
                    ahead = beg[0]
                    ahead = ahead, self.FAP_distance(agent) - self.FAP_distance(ahead)
                else:
                    beg = [el for el in beg_dict[node] if not el.on_shortcut]
                    if beg:
                        ahead = max(beg, key=lambda x: x.position)
                        ahead = ahead, self.FAP_distance(agent) - self.FAP_distance(ahead)
                    else:
                        node = self.graph.adjacency[node][0]
        return ahead


    def wrap_the_simulation(self, initial_conditions):
        reg = {}
        incidents = {}
        t_incidents = 0
        for key, val in self.full_history.items():
            L = list(val['holding_register'].items())
            for el in L:
                if el[0] in reg:
                    reg[el[0]] += len(el[1])
                else:
                    reg[el[0]] = len(el[1])
            inc_L = val['incidents']
            if inc_L:
                incidents[key] = inc_L
                t_incidents += len(inc_L)
        start_reg = [(key,val) for key,val in reg.items() if key in self.graph.starting_nodes]
        start_reg_cum_sum = sum([el[1] for el in start_reg])
        holding_cum_sum = sum(reg.values())
        agents_times = {x: self.agents_off_times.get(x, self.evolutions_number) -\
                           initial_conditions.get(x)[0]\
                        for x in set(self.agents_off_times).union(initial_conditions)}
        cumulative_time = sum(agents_times.values())
        still_on = set(self.agents_off_times).symmetric_difference(initial_conditions)
        self.run_stats = {
            'Time in Holding Cumulative Sum': holding_cum_sum,
            'Time in Holding at Starting Nodes': start_reg_cum_sum,
            'Cumulative Time of Agents in The Network': cumulative_time,
            'Holdings Register': reg,
            'Agents Times in The Network': agents_times,
            'Agents Still in The Network': still_on,
            'Time in Incidents': t_incidents,
            'Incidents Register': incidents,
            'Time in Incidents Alternative': deepcopy(self.inc_count_no_hold),
            'Shortcuts Register': deepcopy(self.shortcuts_register)
        }

    def export_initial_conditions(self):
        export = {}
        for id, ti in self.agents_starting_times.items():
            export[id] = (ti, self.full_history[ti]['agents'][id].procedure)
        return export


class Run:
    """Instance corresponds to particular simulation run.
    Holds information such as initial conditions etc. as well as the results and run's history
    """

    def __init__(self, run_number, initial_conditions, iteration_time, duration_time, environment, history_save,
                 run_stats):
        self.run_number = run_number
        self.initial_conditions = initial_conditions
        self.iteration_time = iteration_time
        self.duration_time = duration_time
        self.environment = environment
        self.history_save = history_save
        self.stats = run_stats
        self.succesfull = None

    def to_pickle(self, path=None, file=None):
        with temp_dir(path):
            if file:
                with open(file + '.pickle', 'wb') as f:
                    pickle.dump(self, f)
            else:
                with open(f'run{self.run_number}' + '.pickle', 'wb') as f:
                    pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            run = pickle.load(f)
        return run
