"""This scripts stores agent class that in the simulation is used to create objects that store information
on the agents participating in the simulation"""
from copy import deepcopy
from re import compile


class Agent(object):
    """Agent class"""

    def __new__(cls, *args, **kwargs):
        """This method was rewritten such that there will be no instances of new agents created if their Id's
        does not fit the right format. However if no id is provided the class is instantiated anyway, this option was
        needed for purpose of pickleing the simulation results."""
        if args:
            try:
                id_format = compile("^[A-Z0-9]{2,3}[0-9]{3,4}$")
                if id_format.match(args[0]) is None:
                    raise ValueError(
                        "Given flight number is  in the wrong format", args[0]
                    )
                else:
                    return object.__new__(cls)
            except ValueError as err:
                print(err.args)
        else:
            return object.__new__(cls)

    def __init__(self, id):
        """Attributes:
        position(list) = [edge start node, edge end node, normalized distance traveled on the edge between (0 to 1]]"""
        # TODO make it all properties
        self._id = id
        self._max_descent_decc_rate = -5
        self._position = None
        self._edge = None
        self._speed = None
        self._acceleration = None
        self._procedure = None
        self._coordinates = None
        self._track = None
        self._next_node_w_limit = None
        self._next_speed_limit = None
        self._distance_to_limit = None
        self._holding = None
        self._in_holding = False
        self.next_holding_node = None
        self._shortcut = None
        self._altitude = None
        self._next_alt_limit = None
        self._next_node_w_alt_limit = None
        self._distance_to_alt_limit = None
        self._otw_nodes = None
        self._on_route = None
        self._adjusted = False
        self.gradient = 0
        self.adjust = False
        self.consecutive_agent = None
        self.ahead_agent = None
        self.adjusted_vox = None
        self.speed_fix = False
        self.hold = False
        self.hold_timer = None
        self.move1 = None
        self.minima_over_5 = True

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls, self.id)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def id(self):
        """Getting Value"""
        return self._id

    @property
    def max_descent_decc_rate(self):
        return self._max_descent_decc_rate

    @property
    def edge(self):
        return self._edge

    @edge.setter
    def edge(self, value):
        self._edge = value

    @property
    def position(self):
        """normalized position on the current edge, ranges between (0, distance to next node)"""
        return self._position

    @position.setter
    def position(self, position):
        """Position is distance to the next viapoint / graph's node"""
        self._position = position

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value

    @property
    def acceleration(self):
        return self._acceleration

    @acceleration.setter
    def acceleration(self, value):
        self._acceleration = value

    @property
    def speed_at_next_node(self):
        return self.get_speed_at_next_node_est(
            distance=self.position,
            distance_to_limit=self.distance_to_limit,
            next_speed_limit=self.next_speed_limit,
        )

    def get_speed_at_next_node_est(self, distance, distance_to_limit, next_speed_limit):
        """Estimates avg speed on the path to the next node in the default setup i.e. when deceleration on the way to
        next node with limit is constant. This procedure is crucial for calculating Separation between agents on
        collisional paths after they (one of them) reaches the intersection"""

        return self.speed + (distance / distance_to_limit) * (
                self.speed - next_speed_limit
        )

    @property
    def procedure(self):
        return self._procedure

    # possibly create a sepecial type of property class that allows to set the procedure attribute once only
    @procedure.setter
    def procedure(self, value):
        self._procedure = value

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = value

    @property
    def track(self):
        return self._track

    @track.setter
    def track(self, value):
        self._track = value

    @property
    def next_node_w_limit(self):
        return self._next_node_w_limit

    @next_node_w_limit.setter
    def next_node_w_limit(self, value):
        self._next_node_w_limit = value

    @property
    def next_speed_limit(self):
        return self._next_speed_limit

    @next_speed_limit.setter
    def next_speed_limit(self, value):
        self._next_speed_limit = value

    @property
    def distance_to_limit(self):
        return self._distance_to_limit

    @distance_to_limit.setter
    def distance_to_limit(self, value):
        self._distance_to_limit = value

    @property
    def holding(self):
        return self._holding

    @holding.setter
    def holding(self, value):
        self._holding = value

    @property
    def shortcut(self):
        return self._shortcut

    @shortcut.setter
    def shortcut(self, value):
        self._shortcut = value

    @property
    def in_holding(self):
        if self.hold_timer:
            return True
        else:
            return False

    @property
    def on_shortcut(self):
        if self._shortcut:
            return True
        else:
            return False

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, value):
        self._altitude = value

    @property
    def next_alt_limit(self):
        return self._next_alt_limit

    @property
    def next_node_alt_limit(self):
        return self._next_node_alt_limit

    @property
    def distance_to_alt_limit(self):
        return self._distance_to_alt_limit

    @distance_to_alt_limit.setter
    def distance_to_alt_limit(self, value):
        self._distance_to_alt_limit = value

    @property
    def otw_nodes(self):
        return self._otw_nodes

    @otw_nodes.setter
    def otw_nodes(self, value):
        self._otw_nodes = value

    @property
    def on_route(self):
        return self._route

    @on_route.setter
    def on_route(self, value):
        self._route = value

    @property
    def adjusted(self):
        return self._adjusted

    @adjusted.setter
    def adjusted(self, value):
        self._adjusted = value
