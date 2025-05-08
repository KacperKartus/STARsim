"""Example graph"""
import os
import glob
import csv
import pandas as pd
from contextlib import contextmanager as ctx
import json
import re
from geopy.distance import geodesic as geopy_gdsc
from geographiclib import geodesic as geolib_gdsc


def to_decimal(s):
    """function recalculating coordinates degrees, minutes, second format to decimal"""
    degrees, minutes, seconds, direction = re.split('[°\'"’”]+', s)
    dec = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    if direction in ('S','W'):
        dec*= -1
    return dec


@ctx
def cwd(path):
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class BaseGraph:
    """This class allows to create representations of STAR procedures routes. In particular by BaseGraph we refer to the
    original directed graph from the aviation national authorities (pansa) or ICAO documents, usually an inverted tree
    (no holdings, nor shortcuts included), where nodes are interpreted as via points of the procedures, an edges are
    tracks in between them. Other information to be stored are, max speed limit on via points, distances between
    via points, information on which via points allow for holdings and how long should they last.

    an inverted tree (follows from the .
    This particular class is created to encode only a base structure with no shortcuts or holdings
    provided this means we expect no more than one edge coming out of a single node."""
    def __init__(self, elevation=0, qnh=1013.25, temp=15, runway_direction=331.8, airport_coordinates=None):
        self.inverted_adjacency = {}
        self.adjacency = dict()
        self.master_dict = dict()
        self._elevation = elevation
        self._qnh = qnh
        self._temperature = temp
        self._airport_coordinates = airport_coordinates
        self._runway_direction = runway_direction
        self.circuit_procedures_dict = dict()
        self.shortcut_dict = dict()
        self.final_conflict_resolving_dict = dict()
        self.long_final_nodes = set()
        self.final_node = None
        self.intersections_set = {}
        self.temp_shortcuts = {}

    def prepare_for_simulation(self):
        self._add_intersections()
        self._add_circuit_procedures()

    @property
    def airport_coordinates(self):
        return self._airport_coordinates

    def add_node(self, node, speed=None, alt=None, coords = None):
        '''Adds node as a key speed as a value to the node_dict
        Args:
            node (str): node's name
            speed (int): speed in [kt] (nm/h (nautical miles per hour)
        '''

        if node not in self.master_dict.keys():
            self.master_dict[node] = {}

        if speed is None or pd.isnull(speed):
            pass
        else:
            self.master_dict[node]['speed'] = speed

        if alt is None or pd.isnull(alt):
            pass
        else:
            self.master_dict[node]['agl'] = self.calculate_agl(alt)

        if coords is None or pd.isnull(coords):
            pass
        else:
            self.master_dict[node]['coordinates'] = coords

    def add_edge(self, node, child=None, track=None, weight=None):
        """Adds edge to adjacency dictionary

        Args:
            node(str): starting node's name
            child(str): ending node's name if no child given we assume and of graph.
            weight(float): weight associated with particular edge, equal to the distance in [nm] between two via points
                            given in the procedure, being represented by the nodes.
            track(float): It is bearing needed to be taken to fly towards the child node.
        """
        if pd.isnull(child):
            child = 'EOG'
            weight = None
            self.final_node = node

        if track is None:
            pass
        else:
            track = float(track)

        if weight is None:
            pass
        else:
            weight = float(weight)

        self.adjacency[node] = (child, weight, track)


    @classmethod
    def from_csv(cls, path_to_folder='.', elevation=0, qnh=1013.25, temp=15):
        """When procedures are provided in the csv form like in the official documents it allows you
         to create the class instance that holds all neccessary information for the simulation. More in README, example
         in data/examples"""
        new = cls(elevation, qnh, temp)
        new.procedures_dict = dict()

        with cwd(path_to_folder):
            for path in glob.glob(os.path.join('procedures','*.csv')):
                df = pd.read_csv(path, index_col=0)
                new.procedures_dict[os.path.basename(path).split('.')[0]] = df

            coords_df = pd.read_csv('COORD.csv', index_col=0)
            coords_df = coords_df.applymap(to_decimal)
            coords_df = coords_df.apply(tuple, axis=1)
            pd.DataFrame(coords_df).apply(lambda row: new.add_node(node=row.name, coords=row[0]), axis=1)
            new.shortcuts_from_file()
            new.final_conflict_resolving_from_csv()
            new.final_nodes_from_csv()
        # Adding nodes to MasterDict
        for key in new.procedures_dict:
            new.procedures_dict[key].apply(lambda row: new.add_node(node=row['TERMINATOR_WAYPOINT'], speed=row['SPEED'],\
                                                   alt=row['ALTITUDE']), axis=1)

            new.procedures_dict[key].apply(lambda row: new.add_edge(node = row['TERMINATOR_WAYPOINT'],
                                                            child = new.procedures_dict[key]['TERMINATOR_WAYPOINT'].\
                                                            shift(-1).loc[row.name],
                                                            weight = new.procedures_dict[key]['DISTANCE'].\
                                                            shift(-1).loc[row.name],
                                                            track = new.procedures_dict[key]['TRACK'].\
                                                            shift(-1).loc[row.name]\
                                                             ), axis=1)
            new.master_dict[new.final_node]['speed'] = 200
        new.starting_nodes = [el[:-2] for el in list(new.procedures_dict.keys())]

        for node in new.adjacency.keys():
            new._add_distance_to_final(node)

        return new

    def _add_distance_to_final(self, node):
        total = 0
        original_node = node

        while node != 'EOG':
            child = self.adjacency[node]
            node, distance = child[0:2]

            if node == 'EOG':
                continue
            else:
                total += distance

        self.master_dict[original_node]['FAP_distance'] = total

    def _add_intersections(self):
        value_count = dict()
        self.intersections_set = {self.final_node}
        for key, value in self.adjacency.items():
            if value[0] in value_count:
                value_count[value[0]] += 1
            else:
                value_count[value[0]] = 1

        for key, value in value_count.items():
            if value > 1:
                self.intersections_set.add(key)

    def _add_circuit_procedures(self):
        right = {}

            # set(input(f'From given list: {self.procedures_dict.keys()} name right circuit procedures (when '
            #               f'providing the input separate names with a comma)').translate({ord(i): None for i in "' "}).split(
            # ','))
        left = {}
            # {procedures for procedures in self.procedures_dict.keys() if procedures not in right}
        self.circuit_procedures_dict = dict(left=left, right=right)

    def final_conflict_resolving_from_csv(self, path_to_folder='.'):
        with open(os.path.join(path_to_folder,'Final_conflicts_resolving.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                self.final_conflict_resolving_dict[row[0]] = tuple(row[1:])

    def final_nodes_from_csv(self, path_to_folder='.'):
        with open(os.path.join(path_to_folder, 'final.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                else:
                    self.long_final_nodes.add(*row)

    def shortcuts_from_file(self):
        """This method adds shortcuts saved in excel file to the shortcut_dict
        """
        df = pd.read_csv('shortcuts.csv', index_col=0)
        df.where(pd.notnull(df), None)
        self.shortcut_dict = dict(df.where(pd.notnull(df), None).iterrows())

    def create_temporary_edge(self, node_name, coords, altitude, destination_node):
        """Creating temporary edges for the purpose of shortcuts"""
        geo_dict = geolib_gdsc.Geodesic.WGS84.Inverse(*coords, *self.master_dict[destination_node]['coordinates'])
        self.master_dict[node_name] = dict(coordinates=coords, agl=altitude)
        # geo_dict results are in meters and azimuth in (-180,180] range, below we switch to our default nautical and
        # [0,360) range
        self.adjacency[node_name] = (destination_node, geo_dict['s12'] * 0.000539957, geo_dict['azi1'] % 360)

    def add_holding(self, node_name, duration):
        self.adjacency[node_name+'_holding'] = (node_name, duration)

    def calculate_agl(self, alt):
        """Recalculating altitude to height above the datum in particular the aerodrome elevation"""
        if alt.startswith('FL'):
            alt = int(alt[2:]) * 100
            baro = alt + (self._qnh - 1013.25)*30
            agl = (4*(baro/1000)*(self._temperature - 15)) + baro - self._elevation
        else:
            agl = int(alt[3:])

        return agl

    def print_adjacency(self):
        for key in self.adjacency.keys():
            print("node", key, ": ", self.adjacency[key])

    def create_inverted_adjacency_register(self):
        for key, value in self.adjacency.items():
            if value[0] in self.inverted_adjacency:
                self.inverted_adjacency[value[0]].append(key)
            else:
                self.inverted_adjacency[value[0]] = [key]

    def count_back(self, node):
        l = len(graph.inverted_adjacency[node])
        dist = 0
        while l == 1:
            node = graph.inverted_adjacency[node][0]
            l = len(graph.inverted_adjacency[node])
            dist += graph.adjacency[node][1]
        return node, dist

# graph = BaseGraph(5)
#
# graph.add_edge('A', 'D', 25)
# graph.add_edge('B', 'D', 20)
# graph.add_edge('C', 'E', 15)
# graph.add_edge('D', 'E', 12)
# graph.add_edge('E', 'F', 7)
#
# graph.print_adjacency()
# graph.add_node('B',190)
# graph.add_node('F',100)
# graph.add_node('E', 200)
# graph._check_next('B',0)

# graph = BaseGraph.from_csv(os.path.expanduser('~/Desktop/PracMag/data'))
